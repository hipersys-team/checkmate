import torch
from torch import Tensor
from torch.utils.data import dataset
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TorchTextDataloader():
    
    def __init__(self, dataset=PennTreebank):
        self.dataset = dataset
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = self.build_vocab()
        train_iter, val_iter, test_iter = self.dataset()
        self.train_data = self.data_process(train_iter)
        self.val_data = self.data_process(val_iter)
        self.test_data = self.data_process(test_iter)

    def build_vocab(self):
        train_iter = self.dataset(split='train')
        vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def data_process(self, raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    
    def get_train_data(self):
        return self.train_data
    
    def get_val_data(self):
        return self.val_data
    
    def get_test_data(self):
        return self.test_data
    
    def vocab_size(self):
        return len(self.vocab)


class FakeDataloader:
    def __init__(self, vocab_sz=50000, train_sz=50000, val_sz=5000, test_sz=5000):
        self.vocab_sz = vocab_sz
        # Generate the data
        self.train_data = self.generate_data(train_sz)
        self.val_data = self.generate_data(val_sz)
        self.test_data = self.generate_data(test_sz)

    def generate_data(self, size):
        # Generate random data
        data = torch.randint(low=0, high=self.vocab_sz, size=(size,), dtype=torch.long)
        return data

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data
    
    def vocab_size(self):
        return self.vocab_sz

        