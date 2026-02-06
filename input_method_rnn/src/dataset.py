"""
定义数据集
"""
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from config import PROCESSED_DATA_DIR, BATCH_SIZE

class InputMethodDataset(Dataset):

    def __init__(self, path):
        self.data = pd.read_json(path, orient="records", lines=True).to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]["input"], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]["target"], dtype=torch.long)
        return input_tensor, target_tensor

def get_dataloader(train=True):
    dataset = InputMethodDataset(PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl"))
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader), len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape, target_tensor.shape)
        break