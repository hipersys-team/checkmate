# Setup a pytorch DDP for model training
# Usage: python benchmarks/run.py --model resnet50 --batch-size 32 --epochs 10 --gpus 2

import torchtext

torchtext.disable_torchtext_deprecation_warning()

import os
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import AdamW
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
from torchtext.datasets import SST2
from torch.utils.data import Dataset, random_split

import argparse
import sys

torchtext.disable_torchtext_deprecation_warning()
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_path)

from common import Trainer, TrainerConfig


def get_pretrained_model():
    from transformers import (
        RobertaForMaskedLM,
        RobertaConfig,
    )

    config_tmp = RobertaConfig.from_pretrained("roberta-base")
    model = RobertaForMaskedLM(config_tmp).cuda()
    return model


def get_model():
    num_classes = 2
    input_dim = 768
    classifier_head = RobertaClassificationHead(
        num_classes=num_classes, input_dim=input_dim
    )
    model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
    return model


def apply_transform(x):
    import torchtext.transforms as T
    from torch.hub import load_state_dict_from_url

    padding_idx = 1
    bos_idx = 0
    eos_idx = 2
    max_seq_len = 256
    xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
    xlmr_spm_model_path = (
        r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
    )
    text_transform = T.Sequential(
        T.SentencePieceTokenizer(xlmr_spm_model_path),
        T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
        T.Truncate(max_seq_len - 2),
        T.AddToken(token=bos_idx, begin=True),
        T.AddToken(token=eos_idx, begin=False),
    )
    # text_transform = XLMR_BASE_ENCODER.transform()
    return text_transform(x[0]), x[1]


def get_data(batch_size) -> tuple[Dataset, Dataset]:
    train_datapipe = SST2(split="train")
    dev_datapipe = SST2(split="dev")

    train_datapipe = train_datapipe.map(apply_transform)
    train_datapipe = train_datapipe.batch(batch_size)
    train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
    train_len = int(len(train_datapipe) * 0.90)
    train_set, test_set = random_split(
        train_datapipe, [train_len, len(train_datapipe) - train_len]
    )
    return train_set, test_set


def get_training_args(args):
    model = get_model()
    train_set, test_set = get_data(args.batch_size)
    optimizer = AdamW(model.parameters(), lr=0.01)
    return model, optimizer, train_set, test_set


def setup_training_config(args):
    return TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        data_loader_workers=16,
        grad_norm_clip=0.1,
        snapshot_load="",
        save_every=10,
        use_amp=False,
        snapshot="CHECKFREQ",
    )


def arg_parser():
    parser = argparse.ArgumentParser(description="PyTorch DDP benchmark")
    parser.add_argument("--model", default="RoBERTa", type=str, help="model")
    parser.add_argument("--batch-size", default=128, type=int, help="batch size")
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus")
    return parser.parse_args()


def main():
    args = arg_parser()
    print(f"Running DDP benchmark with args: {args}")
    model, optimizer, train_data, test_data = get_training_args(args)
    trainer_cfg = setup_training_config(args)

    trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)
    trainer.train()


if __name__ == "__main__":
    main()
