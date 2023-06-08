from torch.utils.data import Dataset

from src.commons.baby_dataset import BabyDataset

datasets = [BabyDataset, ]


def concatenate_datasets(datasets):
    # Load train, val, and test splits from each dataset
    train_data = []
    val_data = []
    test_data = []

    for index, dataset in enumerate(datasets):
        train_dataset, val_dataset, test_dataset = dataset()
        train_dataset = list(train_dataset)
        print(index, len(train_dataset))
        train_data.extend(train_dataset)
        val_data.extend(val_dataset)
        test_data.extend(test_dataset)

    return train_data, val_data, test_data


class CustomDataset(Dataset):
    def __init__(self, data_iter, dataset_size=None):
        self.data = list(data_iter)[:dataset_size] if dataset_size else list(data_iter)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_iter, val_iter, test_iter = CustomDataset(train_data), CustomDataset(val_data), CustomDataset(test_data)

print("DATA LEN ", len(train_iter))
print(train_iter.data[len(train_iter) - 1])

train_data = data_process(train_iter.data)
val_data = data_process(val_iter.data)
test_data = data_process(test_iter.data)
