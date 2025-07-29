import argparse

import os
import time
import torch
import network
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import signal
from torch.distributed._composable.replicate import replicate

from llama.llama_model import ModelArgs, Transformer, EmbeddingStem
from llama.llama_config import all_models
from llama import pipeline

import functools
from torch.optim.lr_scheduler import LambdaLR

torch.manual_seed(0)
torch._dynamo.config.optimize_ddp = "ddp_optimizer"
signal.signal(signal.SIGINT, signal.SIG_DFL)

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from utils import ZeroRedundancyOptimizer


def apply_compile(model: torch.nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.layers.register_module(layer_id, transformer_block)

    print("Compiling each TransformerBlock with torch.compile")


def linear_warmup_linear_decay(
    warmup_steps: int, decay_steps: int, current_step: int
) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_steps + 1))

    else:
        # linear decay
        normalized_step = decay_steps - (current_step - warmup_steps)
        curr_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

    return curr_adjustment


def build_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Build a linear warmup and linear decay scheduler"""
    warmup_steps = warmup_steps
    decay_steps = float(max(1, total_steps - warmup_steps))
    lr_lambda = functools.partial(linear_warmup_linear_decay, warmup_steps, decay_steps)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return warmup_scheduler


class Trainer:
    def __init__(self, config):
        init_process_group()
        self.dtype = torch.bfloat16
        self.model_config = all_models[config.model_type][config.config]
        self.pp_size = config.pp_size
        self.pp_rank = config.pp_rank

        self.raw_model = Transformer.from_model_args(self.model_config).to(
            "cpu", dtype=self.dtype
        )
        if self.pp_size == 1:
            self.model = self.raw_model
        else:
            # pipeline split the model
            pp_parts = pipeline.pipeline_llama(
                self.raw_model, self.pp_size, self.pp_rank, self.model_config
            )
            assert len(pp_parts) == 1
            self.model = pp_parts[0]
        apply_compile(self.model)
        replicate(self.model, bucket_cap_mb=100)

        self.config = config
        self.criterion = torch.nn.functional.cross_entropy
        self.grad_buckets = []
        self.model_size = 0
        for param in self.model.parameters():
            self.model_size += param.element_size() * param.nelement()
        self.network = None
        self.total_fetch_time = 0
        self.total_opt_time = 0

    def thread_setup(self):
        import psutil

        total_threads = psutil.cpu_count(logical=False)
        network_threads = 10  # For Network and other threads
        opt_threads = total_threads - network_threads
        torch.set_num_threads(opt_threads)
        torch.set_num_interop_threads(opt_threads)
        t_list = [t for t in range(network_threads, total_threads)]
        os.sched_setaffinity(0, t_list)

    def generate_fake_input(self):
        first_layer = next(self.model.children())
        if isinstance(first_layer, torch.nn.modules.container.ModuleDict):
            dummy_input = torch.rand(
                (1, self.model_config.max_seq_len, self.model_config.dim), dtype=torch.bfloat16
            )
        else:
            dummy_input = torch.randint(
                0, self.model_config.vocab_size, (1, self.model_config.max_seq_len)
            )
            # raise ValueError(f"Invalid first layer type: {type(first_layer)}")
        return dummy_input

    def setup_gradbucket(self):
        # self.thread_setup()
        sample_input = self.generate_fake_input()
        out = self.model(sample_input)
        loss = self.criterion(out.flatten(), out.flatten())
        loss.backward(retain_graph=True)
        self.model(sample_input)

        self.ddp_ref = replicate.state(self.model)._ddp_weakref()
        self.grad_buckets = self.ddp_ref.reducer._get_grad_buckets()
        bucket_sizes = [
            b.buffer().nelement() * b.buffer().element_size() for b in self.grad_buckets
        ]
        assert sum(bucket_sizes) == self.model_size
        print(f"Gradbucket sizes: {bucket_sizes}")

        dtype_size = self.grad_buckets[0].buffer().element_size()
        print(f"Data type size: {dtype_size}")
        self.nnodes = torch.distributed.get_world_size()
        self.node_rank = torch.distributed.get_rank()
        print(f"Number of nodes: {self.nnodes}, Rank: {self.node_rank}")
        self.network = network.Server(
            41000, bucket_sizes, dtype_size, self.nnodes, self.node_rank
        )

        # Gradbuckets should be ready now
        params = [[] for _ in range(self.nnodes)]
        bucket_count = [0 for _ in range(self.nnodes)]
        assert len(self.grad_buckets) > 1
        for gradbucket in self.grad_buckets:
            index = gradbucket.index()
            param_index = index % self.nnodes
            bucket_count[param_index] += 1
            params[param_index] += gradbucket.parameters()
        print(f"Number of gradbuckets: {bucket_count[self.node_rank]}")

        # https://pytorch.org/tutorials/recipes/compiling_optimizer_lr_scheduler.html
        self.optimizer = ZeroRedundancyOptimizer(
            params=self.ddp_ref.parameters(),
            optimizer_class=optim.AdamW,
            lr=torch.tensor(3e-4),  # Base learning rate
            betas=(0.9, 0.999),  # Default values for AdamW
            eps=1e-8,  # Small value to prevent division by zero
            weight_decay=0.05,  # Regularization strength
            params_per_rank=params,
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer, self.config.lr_warmup_iter, self.config.iterations
        )
        assert self.optimizer.initialized == True

        def optimizer_and_scheduler_step():
            self.optimizer.step()
            self.scheduler.step()

        self.compiled_step = torch.compile(
            optimizer_and_scheduler_step, fullgraph=False, mode="reduce-overhead"
        )
        torch._logging.set_logs(recompiles=True)

    def _update_gradients(self, buffer: torch.Tensor, index: int, is_last: bool):
        bucket_size = buffer.nelement() * buffer.element_size()
        self.network.update_grad_bucket(index, buffer.data_ptr(), bucket_size)
        # buffer.zero_()

    def _fetch_gradients(self):
        for b in self.grad_buckets:
            if b.index() % self.nnodes == self.node_rank:
                self._update_gradients(b.buffer(), b.index(), b.is_last())

    def train(self):
        self.setup_gradbucket()
        self.optimizer.zero_grad(set_to_none=False)
        self.compiled_step()
        print("Starting training")

        for epoch in range(self.config.iterations):
            start = time.time()

            # Note: Optimizers have a different behavior if the gradient is 0 or
            # None (in one case it does the step with a gradient of 0 and in the
            # other it skips the step altogether).
            # self.optimizer.zero_grad(set_to_none=False)
            self._fetch_gradients()
            fetch_end = time.time()
            self.compiled_step()
            opt_end = time.time()

            # print optimizer and fetch time
            fetch_time = (fetch_end - start) * 1000
            opt_time = (opt_end - fetch_end) * 1000
            print(
                f"Iteration {epoch}: Fetch time: {fetch_time:.2f} ms, Opt time: {opt_time:.2f} ms"
            )
            if epoch > 0:
                self.total_fetch_time += fetch_time
                self.total_opt_time += opt_time

    def __del__(self):
        print(
            f"Avg fetch time: {self.total_fetch_time/(self.config.iterations -1):.2f} ms, \
            Avg opt time: {self.total_opt_time/(self.config.iterations - 1):.2f} ms"
        )
        torch.save(self.model.state_dict(), "model_resnet.pth")
        destroy_process_group()


def arg_parser():
    parser = argparse.ArgumentParser(
        description="llama and GPT storage server for torchtitan"
    )
    parser.add_argument(
        "--model-type", default="llama2", type=str, help="class of model (llama2 or 3)"
    )
    parser.add_argument("--config", default="7B", type=str, help="model variant")
    parser.add_argument(
        "--pp-size", default=1, type=int, help="number of pipeline parllelism ranks"
    )
    parser.add_argument(
        "--pp-rank", default=0, type=int, help="pipeline rank this node corresponds to"
    )
    parser.add_argument(
        "--iterations", default=199, type=int, help="number of iterations"
    )
    parser.add_argument(
        "--lr-warmup-iter",
        default=20,
        type=int,
        help="learning rate warm up iteration",
    )
    return parser.parse_args()


def main():
    args = arg_parser()
    print(f"Running DDP benchmark with args: {args}")
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
