#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ckpt

sleep 3
SNAP=DISABLED

source /data/frankwwy/innet_ckpt/script/set_ib_env.sh

# resnet50
/home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=12 --nproc-per-node=1 --node-rank=$MY12RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=saria ./run.py --model resnet50 --batch-size 384 --run-up-to-iter 1000  --lr-warmup-iter 100 --snapshot-policy $SNAP --save-to-file resnet50-ib

# resnet152
/home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=12 --nproc-per-node=1 --node-rank=$MY12RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=saria ./run.py --model resnet152 --batch-size 192 --run-up-to-iter 1000  --lr-warmup-iter 100 --snapshot-policy $SNAP --save-to-file resnet152-ib

# vit
/home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=12 --nproc-per-node=1 --node-rank=$MY12RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=saria ./run.py --model vit_h_14 --batch-size 32 --run-up-to-iter 1000  --lr-warmup-iter 100 --snapshot-policy $SNAP --save-to-file vit-h-14-ib
