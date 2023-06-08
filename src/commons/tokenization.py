from typing import Callable, Optional, List

from torch import Tensor, long, tensor, cat
from torch.utils.data import IterableDataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab


def build_vocabulary(corpus: List[str], tokenizer: Optional[Callable] = None):
    if tokenizer is None:
        tokenizer = get_tokenizer('basic_english')

    vocab = build_vocab_from_iterator(map(tokenizer, corpus), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    return vocab, tokenizer


def data_process(raw_text_iter: IterableDataset, tokenizer: Callable, vocab: Vocab) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [tensor(vocab(tokenizer(item)), dtype=long) for item in raw_text_iter]
    return cat(tuple(filter(lambda t: t.numel() > 0, data)))



