import time
import sys
import os
import network

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

from omegaconf import DictConfig, OmegaConf
import hydra
import signal

torch.manual_seed(0)
signal.signal(signal.SIGINT, signal.SIG_DFL)

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_path)
from utils import ZeroRedundancyOptimizer

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
sys.path.append(parent_path)
from mingpt.model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from common import TrainerConfig


class Trainer:
    def __init__(self, gpt_cfg, opt_cfg, trainer_cfg):
        init_process_group()
        model = GPT(gpt_cfg)
        model.to(torch.bfloat16)
        self.model_size = 0
        for param in model.parameters():
            self.model_size += param.element_size() * param.nelement()
        print(f"Model size: {self.model_size}")
        self.model = DDP(model, gradient_as_bucket_view=True, static_graph=False)
        self.optimizer = ZeroRedundancyOptimizer(
            params=self.model.parameters(),
            optimizer_class=optim.AdamW,
            lr=opt_cfg.learning_rate,
        )
        self.config = trainer_cfg
        self.criterion = torch.nn.CrossEntropyLoss()
        self.grad_buckets = []
        self.network = None


    def get_sample_input(self):
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        sample_text = "Hello, how are you doing today?"
        tokens = tokenizer.encode(sample_text, return_tensors="pt")
        input_size = (self.config.batch_size, tokens.size(1))
        sample_input = tokens.expand(input_size)
        return sample_input

    def thread_setup(self):
        import psutil

        total_threads = psutil.cpu_count(logical=False)
        network_threads = 10  # For Network and other threads
        opt_threads = total_threads - network_threads
        torch.set_num_threads(opt_threads)
        torch.set_num_interop_threads(opt_threads)
        t_list = [t for t in range(network_threads, total_threads)]
        os.sched_setaffinity(0, t_list)

    def setup_gradbucket(self):
        self.thread_setup()
        self.grad_buckets = self.model.reducer._get_grad_buckets()

        sample_input = self.get_sample_input()
        out = self.model(sample_input)
        loss = self.criterion(out, out)
        loss.backward(retain_graph=True)
        self.model(sample_input)

        # This is the order in which the gradients are reduced.
        self.grad_buckets = self.model.reducer._get_grad_buckets()
        bucket_sizes = [
            b.buffer().nelement() * b.buffer().element_size() for b in self.grad_buckets
        ]
        assert sum(bucket_sizes) == self.model_size
        print(f"Gradbucket sizes: {bucket_sizes}")

        dtype_size = self.grad_buckets[0].buffer().element_size()
        self.nnodes = torch.distributed.get_world_size()
        self.node_rank = torch.distributed.get_rank()
        print("Data type size: ", dtype_size)
        print(f"Number of nodes: {self.nnodes}, Rank: {self.node_rank}")
        self.network = network.Server(
            41000, bucket_sizes, dtype_size, self.nnodes, self.node_rank
        )

    def _update_gradients(self, buffer: torch.Tensor, index: int, is_last: bool):
        bucket_size = buffer.nelement() * buffer.element_size()
        self.network.update_grad_bucket(index, buffer.data_ptr(), bucket_size)

    def _fetch_gradients(self):
        for b in self.grad_buckets:
            self._update_gradients(b.buffer(), b.index(), b.is_last())

    def train(self):
        self.setup_gradbucket()
        self.optimizer.zero_grad(set_to_none=False)

        for epoch in range(self.config.max_epochs):
            start = time.time()
            self._fetch_gradients()
            self.optimizer.step()
            end = time.time()
            print(f"Iteration {epoch} took {(end-start) * 1000} ms")


@hydra.main(version_base=None, config_path="../../config/gpt", config_name="config")
def main(cfg: DictConfig):
    gpt_cfg = GPTConfig(**cfg["model"]["gpt_config"])
    opt_cfg = OptimizerConfig(**cfg["optimizer_config"])
    trainer_cfg = TrainerConfig(**cfg["trainer_config"])
    print(gpt_cfg, trainer_cfg)
    trainer = Trainer(gpt_cfg, opt_cfg, trainer_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
    destroy_process_group()
