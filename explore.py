import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter

from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

DO_TRAIN = True
DATASET_SIZE = 5000
PATH = "model.pt"

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = list(data)[:DATASET_SIZE]
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = self.data[index]
        tokens = self.tokenizer(text)
        return tokens

    def __len__(self):
        return len(self.data)

train_iter, val_iter, test_iter = WikiText2()

train_iter, val_iter, test_iter = CustomDataset(train_iter), CustomDataset(val_iter), CustomDataset(test_iter)

dataset = train_iter

# Token analysis
all_tokens = []
for i in range(len(dataset)):
    tokens = dataset[i]
    all_tokens.extend(tokens)

token_counts = Counter(all_tokens)

# Print token frequencies
for token, count in token_counts.most_common(10):
    print(f"Token {token} => {count}")
