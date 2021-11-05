from torch.utils.data.dataloader import DataLoader
from Transformer.data import LanguagePairDataset, MonolingualDataset, WordDict
import os.path as path
import pickle


def collate_fn(samples):  # form samples into batches
    ...


def batch_by_size(data):  # prepare batch sampler
    ...


def prepare_monolingual_dataset(data_path, lang, split, target):
    with open(path.join(data_path, f"{split}.{lang}"), "r", encoding="utf-8") as fl:
        data = pickle.load(fl)

    word_dict = WordDict(path.join(data_path, "dict.txt"))

    return MonolingualDataset(word_dict, data, target)


def prepare_language_pair_dataset(data_path, src_lang, tgt_lang, split):

    src = prepare_monolingual_dataset(data_path, src_lang, split, target="present")
    tgt = prepare_monolingual_dataset(data_path, tgt_lang, split, target="future")

    return LanguagePairDataset(src, tgt)


def prepare_dataset_itr(dataset):
    return DataLoader(dataset, batch_sampler=None, collate_fn=collate_fn)

