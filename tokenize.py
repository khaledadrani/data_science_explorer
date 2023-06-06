import torch
from torchtext.datasets import dataset, WikiText2
from transformers import GPT2Tokenizer

def data_process(raw_text_iter: dataset.IterableDataset, tokenizer: GPT2Tokenizer) -> torch.Tensor:
    """Converts raw text into a flat Tensor using GPT-2 tokenizer."""
    data = [torch.tensor(tokenizer.encode(item), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter = WikiText2(split='train')
# Example usage
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
processed_data = data_process(list(train_iter)[:10], tokenizer)

print(processed_data[0])
