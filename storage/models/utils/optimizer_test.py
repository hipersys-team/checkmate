import torch
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from optimizer import ZeroRedundancyOptimizer
import time


def test():
    init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"Rank {rank} of {world_size}")
    model = models.resnet50().to("cpu")
    ddp = DDP(model, gradient_as_bucket_view=True, static_graph=False)

    input = torch.randn(128, 3, 224, 224)
    out = ddp(input)
    out.sum().backward(retain_graph=True)
    out = ddp(input)

    # Gradbuckets should be ready now
    params = [[] for _ in range(world_size)]
    gradbuckets = ddp.reducer._get_grad_buckets()
    assert len(gradbuckets) > 1
    for gradbucket in gradbuckets:
        index = gradbucket.index()
        param_index = index % world_size
        params[param_index] += gradbucket.parameters()

    optimizer = ZeroRedundancyOptimizer(
        ddp.parameters(),
        optimizer_class=torch.optim.SGD,
        lr=0.01,
        parameters_as_bucket_view=True,
        params_per_rank=params,
    )

    for _ in range(10):
        start = time.time()
        optimizer.zero_grad(set_to_none=False)
        optimizer.step()
        print(f"Time taken for one step {(time.time() - start) * 1000}ms")

    optimizer.consolidate_state()
    if rank == 0:
        app_state = {
            "optimizer": optimizer.state_dict(),
            "model": ddp.state_dict(),
        }
        torch.save(app_state, f"checkpoint_{rank}.pth")
    print(
        f"Total number of parameters {sum([g.buffer().numel() for g in gradbuckets])}"
    )
    print(f"Number of elements on Rank {rank} {len(optimizer._buckets[0][rank])}")
    print(f"Number of grad buckets {len(ddp.reducer._get_grad_buckets())}")


if __name__ == "__main__":
    test()
    destroy_process_group()
