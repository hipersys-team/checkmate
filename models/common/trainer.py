"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from dataclasses import dataclass
from typing import Optional
import time
import os
import logging
from common.measurement import MeasurementData
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, get_world_size

import torchsnapshot
from torchsnapshot import StateDict, Snapshot as Snapshot

from checkfreq.cf_checkpoint import CFCheckpoint
from checkfreq.cf_manager import CFManager, CFMode
from checkfreq.cf_iterator import CFIterator

import torch.distributed as dist

import warnings

warnings.filterwarnings("ignore", module="torchsnapshot")


def forge_unique_name(module, prefix=""):
    """
    Generate a unique name for a given torch.nn.Module.

    Args:
        module (torch.nn.Module): The module to generate a unique name for.
        prefix (str): A prefix string to carry hierarchical information.

    Returns:
        str: A unique name for the module.
    """
    if isinstance(module, nn.parallel.DistributedDataParallel):
        # If wrapped by DDP, consider the DDP wrapping and recurse into the module
        ddp_prefix = "DDP"
        return forge_unique_name(module.module, prefix=f"{prefix}.{ddp_prefix}")

    # Build the name based on the class name and prefix
    unique_name = (
        f"{prefix}.{module.__class__.__name__}" if prefix else module.__class__.__name__
    )

    # Include hierarchical information based on submodules
    for name, child in module.named_children():
        unique_name += f".{name}-{forge_unique_name(child)}"

    return unique_name[:64]


def ddp_setup():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        init_process_group(backend="cuda:nccl,cpu:gloo")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        init_process_group(backend="gloo")


def broadcast_file_name(file_name, src=0):
    """
    Broadcasts a file name from the source process to all other processes.

    Args:
        file_name (str): The file name to broadcast (only needed on src process).
        src (int): The rank of the source process.

    Returns:
        str: The broadcasted file name.
    """
    # Encode the file name to bytes
    max_length = 256  # Assume the max file name length
    if dist.get_rank() == src:
        encoded_file_name = file_name.encode("utf-8")
        assert len(encoded_file_name) < max_length, "File name too long!"
        file_name_tensor = torch.zeros(max_length, dtype=torch.uint8)
        file_name_tensor[: len(encoded_file_name)] = torch.tensor(
            list(encoded_file_name), dtype=torch.uint8
        )
    else:
        file_name_tensor = torch.zeros(max_length, dtype=torch.uint8)

    # Broadcast the tensor
    dist.broadcast(file_name_tensor, src=src)

    # Decode the tensor back to a string
    decoded_file_name = bytes(file_name_tensor.tolist()).decode("utf-8").rstrip("\x00")
    return decoded_file_name


@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path_nosave: Optional[str] = None
    snapshot_path_save: Optional[str] = None
    snapshot_load: Optional[str] = None
    save_every: int = None
    use_amp: bool = None
    precision: str = None
    snapshot: str = None
    criterion: Optional[str] = "CrossEntropy"  # todo: add other type (uncommon)
    warmup_iter: int = 10
    run_up_to_iter: int = 20
    save_to_file: Optional[str] = None
    lr_warmup_iter: int = 100


class Trainer:

    def __init__(
        self,
        trainer_config: TrainerConfig,
        model,
        optimizer,
        scheduler,
        train_dataset,
        test_dataset=None,
    ):
        ddp_setup()
        snapshot_sch_logger = logging.getLogger(torchsnapshot.scheduler.__name__)
        snapshot_sch_logger.propagate = False

        self.config = trainer_config
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = dist.get_world_size()
        # data stuff
        self.train_dataset = train_dataset
        self.snapshot_policy = self.config.snapshot
        # initialize train states
        self.epochs_run = 0
        self.iteration = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = self.config.save_every
        self.criterion = nn.CrossEntropyLoss().to(self.local_rank)
        if self.config.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None
        if self.config.precision == "fp16":
            model.to(torch.float16)
        elif self.config.precision == "bf16":
            model.to(torch.bfloat16)
        # wrap with DDP. this step will synch model across all the processes.
        # breakpoint()
        self.model = DDP(
            self.model, device_ids=[self.local_rank], broadcast_buffers=False
        )
        self.cf_manager = None
        if trainer_config.snapshot == "CHECKFREQ":
            self.chk = CFCheckpoint(model=self.model, optimizer=self.optimizer)
            self.cf_manager = CFManager(
                trainer_config.snapshot_path_save, self.chk, scaler=self.scaler
            )
        self.app_state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "extra_state": StateDict(iteration=0, epoch=0),
        }
        # load snapshot if available. only necessary on the first node.
        if self.config.snapshot_load is None:
            self.config.snapshot_load = ""
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = (
            self._prepare_dataloader(test_dataset) if test_dataset else None
        )
        self._load_snapshot()
        self.iteration_compute_time = 0
        self.snapshot_time = 0
        self.warmup_iter = trainer_config.warmup_iter
        self.niters = 0
        self.nsnapshots = 0
        self.run_up_to_iter = trainer_config.run_up_to_iter
        self.last_pending_snapshot = None

        if trainer_config.save_to_file:
            snapshot_str = (
                f"ASYNC{trainer_config.save_every}"
                if trainer_config.snapshot == "ASYNC"
                else (
                    f"SYNC{trainer_config.save_every}"
                    if trainer_config.snapshot == "SYNC"
                    else trainer_config.snapshot
                )
            )
            if self.global_rank == 0:
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                file_name = (
                    f"{trainer_config.save_to_file}_"
                    + f"{trainer_config.batch_size}_"
                    + f"{trainer_config.precision}_"
                    + f"{snapshot_str}_"
                    + f"{ts}_"
                )
            else:
                file_name = ""
            self.train_filename = (
                broadcast_file_name(file_name, src=0) + f"{self.global_rank}.csv"
            )
            self.train_iterlog = open(self.train_filename, "w")
            self.train_iterlog.write("iter,iter_time,wct,loss,nsnapshots\n")
        else:
            self.train_iterlog = None

    def _prepare_dataloader(self, dataset: Dataset):
        # Keep multiprocessing_context="spawn" when using DPDK-based NCCL backend as
        # the dataloader forks multiple processes on each enumerate() call. The fork call
        # conflicts with DPDK runtime and shmem-based queues also break.
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=8,
            sampler=DistributedSampler(dataset),
            # multiprocessing_context="spawn",
        )
        if self.snapshot_policy == "CHECKFREQ":
            loader = CFIterator(
                loader,
                self.config.batch_size * self.world_size,
                arch=f"{forge_unique_name(self.model)}_{self.config.batch_size*self.world_size}",
                worker_id=self.global_rank,
                cf_manager=self.cf_manager,
            )
        return loader

    def _load_snapshot(self):
        try:
            snapshot = Snapshot(path=self.config.snapshot_load)
            snapshot.restore(app_state=self.app_state)
        except:
            print("Snapshot not found. Training model from scratch")
            return

        self.epochs_run = snapshot.read_object(path="0/extra_states/epoch")
        self.iteration = snapshot.read_object(path="0/extra_states/iteration")
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(self.config.use_amp)
        ):
            # starter = torch.cuda.Event(enable_timing=True)
            # ender = torch.cuda.Event(enable_timing=True)
            # starter.record()
            output = self.model(source)
            # ender.record()
            # torch.cuda.synchronize()
            # print(f"fw_time: {starter.elapsed_time(ender)}")
            loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))

        if train:
            # starter = torch.cuda.Event(enable_timing=True)
            # ender = torch.cuda.Event(enable_timing=True)
            # starter.record()
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
                if self.global_rank == 0 and self.cf_manager:
                    self.cf_manager.weight_update()
                else:
                    self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
                if self.global_rank == 0 and self.cf_manager:
                    self.cf_manager.weight_update()
                else:
                    self.optimizer.step()
            self.scheduler.step()
            # ender.record()
            # torch.cuda.synchronize()
            # print(f"bw_time: {starter.elapsed_time(ender)}")
        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        for iteration, (source, targets) in enumerate(dataloader):
            time_start = time.perf_counter_ns()
            if self.niters == self.run_up_to_iter:
                break
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank, non_blocking=True)
            targets = targets.to(self.local_rank, non_blocking=True)
            batch_loss = self._run_batch(source, targets, train)
            time_compute_end = time.perf_counter_ns()
            iter_time = time_compute_end - time_start

            time_compute_end = time.perf_counter_ns()
            if iteration > 0:
                self.iteration_compute_time += iter_time
                self.niters += 1

            if (
                self.save_every is not None
                and self.snapshot_policy != "DISABLED"
                and self.snapshot_policy != "CHECKFREQ"
                and self.global_rank == 0
                and train
                and self.niters > 0
                and (self.niters + 1) % self.save_every == 0
            ):
                curr_time = self._save_snapshot(epoch, iteration)
                self.snapshot_time += curr_time
                self.nsnapshots += 1
            time_end = time.perf_counter_ns()
            print(
                f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iteration} | {step_type} Loss {batch_loss:.5f} | time {(time_end-time_start)/1e9:.5f}"
            )
            if self.train_iterlog:
                self.train_iterlog.write(
                    f"{self.niters},{time_end-time_start},{time_end},{batch_loss},{self.nsnapshots if self.snapshot_policy != 'CHECKFREQ' else self.cf_manager.chk_global_id - 1}\n"
                )
            # if epoch == 0 and self.niters < 3:
            #     time.sleep(1)

    def _save_snapshot(self, epoch, iteration):
        # capture snapshot
        # starter = torch.cuda.Event(enable_timing=True)
        # ender = torch.cuda.Event(enable_timing=True)
        # starter.record()
        time_start = time.perf_counter_ns()
        self.app_state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "extra_state": StateDict(iteration=iteration, epoch=epoch),
        }
        if self.snapshot_policy == "ASYNC":
            self.last_pending_snapshot = Snapshot.async_take(
                path=f"{self.config.snapshot_path_save}/snapshot",
                app_state=self.app_state,
            )
        elif self.snapshot_policy == "SYNC":
            Snapshot.take(
                path=f"{self.config.snapshot_path_save}/snapshot",
                app_state=self.app_state,
            )
        # elif self.snapshot_policy == "CHECKFREQ":
        #     assert self.chk._snapshot(active_snapshot=0) != False
        elif self.snapshot_policy == "TORCH":
            torch.save(
                self.model.state_dict(),
                f"model_{epoch}_{iteration}.pth",
            )

        time_end = time.perf_counter_ns()
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # puck this library
        if self.snapshot_policy == "SYNC":
            shutil.rmtree(f"{self.config.snapshot_path_save}/snapshot")
        return time_end - time_start

    def _warmup(self):
        print("Warming up GPU...")
        dummy_source, _ = next(iter(self.train_loader))

        out = None
        for _ in range(self.warmup_iter):
            out = self.model(dummy_source)
        loss = self.criterion(out, out)
        loss.backward(retain_graph=True)
        out = self.model(dummy_source)
        self.optimizer.zero_grad(set_to_none=False)

        grad_buckets = self.model.reducer._get_grad_buckets()
        initial_bucket_size = [
            b.buffer().nelement() * b.buffer().element_size() for b in grad_buckets
        ]
        self.optimizer.zero_grad(set_to_none=False)
        print("Warming up GPU done.")
        print(f"Element sizes: {grad_buckets[0].buffer().element_size()}")
        print(f"Final bucket sizes: {initial_bucket_size}")

    def train(self):
        # self._warmup()
        for epoch in range(self.epochs_run, self.config.max_epochs):
            if self.niters == self.run_up_to_iter:
                break
            epoch += 1
            self._run_epoch(epoch, self.train_loader, train=True)
            # eval run
            # if self.test_loader:
            #     self._run_epoch(epoch, self.test_loader, train=False)

    def get_measurement(self):
        return MeasurementData(
            model="",
            model_size=0,
            model_params=0,
            world_size=get_world_size(),
            batch_size=self.config.batch_size,
            snap_policy=self.snapshot_policy,
            total_iter_time=self.iteration_compute_time,
            total_snap_time=self.snapshot_time,
            total_iter_cnt=self.niters,
            total_snap_cnt=self.nsnapshots,
            snap_overhead=(
                self.snapshot_time / (self.snapshot_time + self.iteration_compute_time)
            ),
        )

    def get_itertimes(self):
        return self.iteration_compute_time, self.niters

    def get_snapshottimes(self):
        return self.snapshot_time, self.nsnapshots

    def __del__(self):
        if int(os.environ["LOCAL_RANK"]) == 0:
            itertime, itercnt = self.get_itertimes()
            snaptime, snapcnt = self.get_snapshottimes()
            itertime_ms = itertime / 1e6
            snaptime_ms = snaptime / 1e6
            print(
                f"Train iter time total: {itertime_ms}ms, # iterations: {itercnt} ({itertime_ms/itercnt}ms per iteration)"
            )
            if snapcnt > 0:
                print(
                    f"Train snap time total: {snaptime_ms}ms, # snapshots: {snapcnt} ({snaptime_ms/snapcnt}ms per snapshot)"
                )
            print(f"snapshot overhead: {snaptime_ms / (snaptime_ms + itertime_ms)}")

        if self.last_pending_snapshot is not None:
            self.last_pending_snapshot.wait()

        destroy_process_group()
