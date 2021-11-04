from torch.utils.data.dataset import T_co
from Transformer.data import WordDict
from torch.utils.data import Dataset, BatchSampler
import os.path as path
import pickle
import numpy as np


def collate_fn(samples):  # form samples into batches
    ...


def batch_by_size(data: list, strategy="sort"):  # prepare batch sampler
    ...


class MonolingualDataset(Dataset):
    def __init__(self, word_dict: WordDict, data: list, target="future") -> None:
        super().__init__()

        self.data = data
        self.word_dict = word_dict

        assert target in ["future", "past", "present"]

        self.target = target

    def __getitem__(self, index) -> T_co:
        source = self.data[index][:]
        target = self.data[index][:]

        source = [self.word_dict.bos_idx] + source

        if self.target == "past":
            target = [self.word_dict.bos_idx, self.word_dict.bos_idx] + target
            source = source + [self.word_dict.eos_idx]
        elif self.target == "future":
            target = target + [self.word_dict.eos_idx]
        elif self.target == "present":
            target = [self.word_dict.bos_idx] + target + [self.word_dict.eos_idx]
            source = source + [self.word_dict.eos_idx]
        else:
            raise Exception(f"Target type {self.target} is not supported!")

        return {"source": source, "target": target}

    def __len__(self):
        return len(self.data)


def prepare_monolingual_dataset(data_path, lang, split, target):
    with open(path.join(data_path, f"{split}.{lang}"), "rb") as fl:
        data = pickle.load(fl)

    word_dict = WordDict(path.join(data_path, "dict.txt"))

    return MonolingualDataset(word_dict, data, target)
