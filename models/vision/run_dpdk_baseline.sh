#!/bin/bash 

eval "$(conda shell.bash hook)"
conda activate ckpt

sleep 3

source /data/frankwwy/innet_ckpt/scripts/set_dpdk_env.sh

# resnet50
NCCL_BUFFSIZE=524288 sudo -E nice -n -20 numactl -m 0 -C 0-7,24-31  /home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=12 --nproc-per-node=1 --node-rank=$MY12RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=saria ./run.py --model resnet50 --batch-size 384 --run-up-to-iter 1000  --lr-warmup-iter 100 --snapshot-policy DISABLED --save-to-file resnet50-dpdk

sleep 30

# resnet152
NCCL_BUFFSIZE=524288 sudo -E nice -n -20 numactl -m 0 -C 0-7,24-31  /home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=12 --nproc-per-node=1 --node-rank=$MY12RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=saria ./run.py --model resnet152 --batch-size 192 --run-up-to-iter 1000  --lr-warmup-iter 100 --snapshot-policy DISABLED --save-to-file resnet152-dpdk

sleep 30

# vit
NCCL_BUFFSIZE=524288 sudo -E nice -n -20 numactl -m 0 -C 0-7,24-31  /home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=12 --nproc-per-node=1 --node-rank=$MY12RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=saria ./run.py --model vit_h_14 --batch-size 32 --run-up-to-iter 1000  --lr-warmup-iter 100 --snapshot-policy DISABLED --save-to-file vit-h-14-dpdk