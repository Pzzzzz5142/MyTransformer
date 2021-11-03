import torch
from torch.utils.data import Dataset


class MonolingualDataset(Dataset):
    def __init__(self, vocab_size) -> None:
        super().__init__()

