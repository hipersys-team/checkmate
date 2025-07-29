# Vision Model Training

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 run.py
```

or we can provide distributed training args even on a single node:

```bash
torchrun --nnodes=1 --nproc-per-node=1 --node-rank=0 --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=localhost run.py
```

To run it on multiple nodes, simply change the `--nnodes` and `--node-rank` args. For example, to run on 2 nodes with 1 process per node:

```bash
torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=parvati run.py
```

> Change --node-rank=0 to --node-rank=1 on the second node.

Also, set `NCCL_SOCKET_IFNAME` to the local interface.
For other environment variables, please refer to [NCCL docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

## Snapshot Policies

1. SYNC
2. ASYNC
3. CHECKFREQ

Modify `Snapsnot` args for Trainer accordingly.

# MinGPT

```bash
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=sr01 --master_port=1234 run.py --config-name=config --config-path=../../config/gpt model=gpt3_xl trainer_config.snapshot=CHECKFREQ trainer_config.run_up_to_iter=20 trainer_config.save_every=5
```
