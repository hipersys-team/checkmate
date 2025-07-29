"""run.py:"""
#!/usr/bin/env python
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(size):
    """ All-Reduce example."""
    rank = int(os.environ["LOCAL_RANK"])
    world_rank = int(os.environ["RANK"])
    if world_rank == 0:
        output = torch.zeros(size).cuda(rank)
    else:
        output = torch.ones(size).cuda(rank)
    s = torch.cuda.Stream()
    handle = dist.all_reduce(output, async_op=True)
    # Wait ensures the operation is enqueued, but not necessarily complete.
    handle.wait()
    # Using result on non-default stream.
    with torch.cuda.stream(s):
        s.wait_stream(torch.cuda.default_stream())
        print(output)

def init_processes(size):
    dist.init_process_group("nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    run(size)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100000, help="Tensor size.")
    args = parser.parse_args()
    init_processes(args.size)