import timeit
import os
import argparse

import torch
import torchvision.models as models
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from utils import ZeroRedundancyOptimizer


def compiled_optimizer(optimizer):
    @torch.compile(fullgraph=False)
    def step(optimizer):
        optimizer.step()

    step(optimizer)


def _apply_ac_to_transformer_block(module: nn.Module):
    return ptd_checkpoint_wrapper(module, preserve_rng_state=False)


def apply_ac(model: nn.Module):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(transformer_block)
        model.layers.register_module(layer_id, transformer_block)

    print(f"Applied full activation checkpointing to the model")


def bench(model, input, model_name):
    from torch._inductor import config as inductor_config

    init_process_group(backend="gloo")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"Rank {rank} of {world_size}")
    ddp = DDP(model, gradient_as_bucket_view=True, static_graph=False)
    assert torch.get_num_threads() == int(os.environ.get("OMP_NUM_THREADS"))
    inductor_config.cpp.threads = torch.get_num_threads()

    out = ddp(input)
    out.sum().backward(retain_graph=True)
    out = ddp(input)

    # Gradbuckets should be ready now
    params = [[] for _ in range(world_size)]
    gradbuckets = ddp.reducer._get_grad_buckets()
    bucket_count = [0 for _ in range(world_size)]
    assert len(gradbuckets) > 1
    for gradbucket in gradbuckets:
        index = gradbucket.index()
        param_index = index % world_size
        bucket_count[param_index] += 1
        params[param_index] += gradbucket.parameters()
    print(f"Number of gradbuckets: {bucket_count[rank]}")

    optimizer = ZeroRedundancyOptimizer(
        ddp.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=0.01,
        # parameters_as_bucket_view=True,
        params_per_rank=params,
    )

    latencies = []
    compiled_optimizer(optimizer)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=False)
        # randomized_grads(params)
        latency = timeit.Timer(lambda: compiled_optimizer(optimizer)).timeit(number=1)
        latencies.append(latency)
        print(f"Time taken for one step {latency * 1000} ms")

    # Print min, median, max
    latencies.sort()
    min, median, max = latencies[0] * 1000, latencies[5] * 1000, latencies[9] * 1000
    print("\nModel,rank,world_size,threads,min,median,max")
    print(
        f"{model_name},{rank},{world_size},{torch.get_num_threads()},{min:.2f},{median:.2f},{max:.2f}"
    )


def randomized_grads(params):
    for param in params:
        param.grad.copy_(torch.randn_like(param))


def generate_fake_input(model_config, model):
    first_layer = next(model.children())
    if isinstance(first_layer, torch.nn.modules.container.ModuleDict):
        dummy_input = torch.rand((1, model_config.max_seq_len, model_config.dim))
    else:
        dummy_input = torch.randint(
            0, model_config.vocab_size, (1, model_config.max_seq_len)
        )
        # raise ValueError(f"Invalid first layer type: {type(first_layer)}")
    return dummy_input


def get_training_args(args):
    from llama.llama_config import all_models
    from llama.llama_model import Transformer

    if args.model.startswith("gpt") or "B" in args.model:
        model_config = all_models["llama2"][args.model]
        model = Transformer.from_model_args(model_config).to(
            "cpu", dtype=torch.bfloat16
        )
        input = generate_fake_input(model_config, model)
        if args.model == "26B" or args.model == "70B":
            apply_ac(model)
        return model, input
    else:
        model = models.__dict__[args.model]().to("cpu")
        input = torch.randn(4, 3, 224, 224)
        return model, input


def arg_parser():
    parser = argparse.ArgumentParser(description="PyTorch DDP benchmark")
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=[
            "resnet50",
            "resnet152",
            "vgg11",
            "vit_h_14",
            "gpt2",
            "gpt3xl",
            "gpt3_6_7B",
            "7B",
            "13B",
            "26B",
            "70B",
        ],
        type=str,
        help="model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["GLOO_SOCKET_IFNAME"] = "100gp1"
    args = arg_parser()
    model, input = get_training_args(args)
    bench(model, input, args.model)
    destroy_process_group()
