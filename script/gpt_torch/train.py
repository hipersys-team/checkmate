import os
import math
import time
import torch
from torch import nn, Tensor
from torch.utils.data.dataset import IterableDataset
from typing import Tuple

def batchify(data: Tensor, bsz: int, device: str) -> Tensor:
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    return data.view(bsz, seq_len).t().contiguous().to(device)

def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train(model: nn.Module, train_data: Tensor, criterion, optimizer, scheduler, device: str, bptt: int, ntokens: int, epoch: int, log_interval: int = 20) -> None:
    model.train()
    total_loss = 0.
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                   epoch, batch, num_batches, scheduler.get_last_lr()[0],
                   elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor, criterion, bptt: int, ntokens: int, device: str) -> float:
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)
