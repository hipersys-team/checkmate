# Setup a pytorch DDP for model training
# Usage: python benchmarks/run.py --model resnet50 --batch-size 32 --epochs 10 --gpus 2

import os
import torch

from torch.utils.data import random_split

import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
import torchvision.models as models
import argparse
import sys
import random
import numpy as np
import functools
import pathlib

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_path)

from common import Trainer, TrainerConfig

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=False)


def get_model(args):
    return models.__dict__[args.model]()


def get_data(args):
    transform = None
    if args.precision == "fp16":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.half())]
        )
    # elif args.model.startswith("vit"):
    #     transform = transforms.Compose(
    #         [
    #             transforms.Resize(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #         ]
    #     )
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = (
        torchvision.datasets.ImageFolder("/data/dataset/imagenet/train",
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    )
    return train_dataset


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


def get_training_args(args):
    model = get_model(args)
    dataset = get_data(args)
    train_len = int(len(dataset) * 0.90)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,  # Base learning rate
        betas=(0.9, 0.999),  # Default values for AdamW
        eps=1e-8,  # Small value to prevent division by zero
        weight_decay=0.05,  # Regularization strength
    )
    scheduler = build_lr_scheduler(optimizer, args.lr_warmup_iter, args.run_up_to_iter)
    return model, optimizer, scheduler, train_set, test_set


def setup_training_config(args):
    return TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        data_loader_workers=16,
        grad_norm_clip=0.1,
        snapshot_load="",
        use_amp=False,
        precision=args.precision,
        snapshot=args.snapshot_policy,
        snapshot_path_save=args.snapshot_save_dir,
        save_to_file=args.save_to_file,
        run_up_to_iter=args.run_up_to_iter,
        save_every=args.snapshot_freq
    )


def arg_parser():
    parser = argparse.ArgumentParser(description="PyTorch DDP benchmark")
    parser.add_argument("--model", default="resnet50", type=str, help="model")
    parser.add_argument("--batch-size", default=512, type=int, help="batch size")
    parser.add_argument("--epochs", default=10000, type=int, help="number of epochs")
    parser.add_argument(
        "--run-up-to-iter", default=200, type=int, help="number of epochs"
    )
    parser.add_argument(
        "--lr-warmup-iter",
        default=20,
        type=int,
        help="learning rate warm up iteration",
    )
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus")
    parser.add_argument("--precision", default="fp32", type=str, help="precision")
    parser.add_argument(
        "--save-to-file", default=None, type=str, help="save training log"
    )
    parser.add_argument("--snapshot-policy", default="DISABLED", type=str, help="snapshot policy")
    parser.add_argument("--snapshot-save-dir", default="snapshot", type=str, help="snapshot saving directory")
    parser.add_argument("--snapshot-freq", default=10, type=int, help="how often to snapshot")
    return parser.parse_args()


def main():
    args = arg_parser()
    print(f"Running DDP benchmark with args: {args}")
    model, optimizer, scheduler, train_data, test_data = get_training_args(args)
    trainer_cfg = setup_training_config(args)

    trainer = Trainer(trainer_cfg, model, optimizer, scheduler, train_data, test_data)
    trainer.train()


if __name__ == "__main__":
    main()
