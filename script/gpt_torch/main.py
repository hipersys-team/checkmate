import argparse
import torch
from gpt import GPT
from dataloader import FakeDataloader, TorchTextDataloader
from train import train, evaluate, batchify
from torch import nn, optim
import math

def get_dtype(dtype_name):
    if dtype_name == 'float16':
        return torch.float16
    elif dtype_name == 'float32':
        return torch.float32
    elif dtype_name == 'bfloat16':
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

def main():
    parser = argparse.ArgumentParser(description='Train a GPT model.')
    parser.add_argument('--bsz_train', type=int, default=216, help='Training batch size')
    parser.add_argument('--bsz_val', type=int, default=10, help='Validation batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--bptt', type=int, default=256, help='Sequence length')
    parser.add_argument('--fake_ntokens', type=int, default=1000000, help='Number of tokens to generate in fake data') 
    parser.add_argument('--vocabsz', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--emsize', type=int, default=512, help='Embedding size')
    parser.add_argument('--nhid', type=int, default=-4, help='Hidden size')
    parser.add_argument('--nlayers', type=int, default=8, help='Number of layers')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32', 'bfloat16'], help='Data type for model and data')
    parser.add_argument('--use_real_data', action='store_true', help='Use real data instead of fake data')
    parser.add_argument('--train_sz', type=int, default=500000, help='Training size for fake data')
    parser.add_argument('--val_sz', type=int, default=5000, help='Validation size for fake data')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = get_dtype(args.dtype)
    torch.set_default_dtype(dtype)

    if args.use_real_data:
        dataloader = TorchTextDataloader()
    else:
        dataloader = FakeDataloader(args.vocabsz, args.train_sz, args.val_sz)
    args = parser.parse_args()

    if args.nhid < 0:
        args.nhid = -args.nhid * args.emsize
    model = GPT(args.vocabsz, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
                                          
    train_data = batchify(dataloader.get_train_data(), args.bsz_train, device)
    val_data = batchify(dataloader.get_val_data(), args.bsz_val, device)
    
    for epoch in range(1, args.epochs + 1):
        train(model, train_data, criterion, optimizer, scheduler, device, args.bptt, args.vocabsz, epoch)
        val_loss = evaluate(model, val_data, criterion, args.bptt, args.vocabsz, device)
        print('-' * 89)
        print('| end of epoch {:3d} | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, val_loss, math.exp(val_loss)))
        print('-' * 89)
        scheduler.step()

if __name__ == '__main__':
    main()
