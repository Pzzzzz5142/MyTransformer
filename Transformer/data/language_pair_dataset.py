from copy import deepcopy
import numpy as np
from numpy.core.fromnumeric import argsort
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader, IterableDataset
from torch.utils.data.dataset import T_co
from torch.utils.data.sampler import SequentialSampler
from Transformer.data import prepare_monolingual_dataset, MonolingualDataset
from typing import Iterator


class PinMemoryBatch(object):
    def __init__(self, id, input_tokens, output_tokens, target) -> None:
        self.batch = {
            "id": torch.LongTensor(id),
            "net_input": {
                "input_tokens": torch.LongTensor(input_tokens),
                "output_tokens": torch.LongTensor(output_tokens),
            },
            "target": torch.LongTensor(target),
        }

    def pin_memery(self):
        self.batch["id"] = self.batch["id"].pin_memory()
        self.batch["net_input"]["input_tokens"] = self.batch["net_input"][
            "input_tokens"
        ].pin_memory()
        self.batch["net_input"]["output_tokens"] = self.batch["net_input"][
            "output_tokens"
        ].pin_memory()
        self.batch["target"] = self.batch["target"].pin_memory()

    def to(self, device: torch.device):
        self.batch["id"] = self.batch["id"].to(device)
        self.batch["net_input"]["input_tokens"] = self.batch["net_input"][
            "input_tokens"
        ].to(device)
        self.batch["net_input"]["output_tokens"] = self.batch["net_input"][
            "output_tokens"
        ].to(device)
        self.batch["target"] = self.batch["target"].to(device)
        return self

    def get_batch(self):
        return self.batch


def collate_fn(samples: list):  # form samples into batches
    ind = []
    input_tokens = []
    output_tokens = []
    target = []
    for sample in samples:
        ind.append(sample["id"])
        data = sample["data"]
        input_tokens.append(data["src_lang"])
        output_tokens.append(data["tgt_lang"])
        target.append(data["target"])

    return PinMemoryBatch(ind, input_tokens, output_tokens, target)


def batch_by_size(
    src, tgt, max_tokens, strategy="tgt_src", long_first=True
):  # prepare batch sampler

    assert strategy in ["src_tgt", "tgt_src", "shuffle", "src", "tgt"]

    info = {
        "total_padding_num": 0,
        "src_padding_num": 0,
        "tgt_padding_num": 0,
        "batch_num": 0,
    }

    if strategy == "shuffle":
        raise NotImplementedError("Shuffle is not implemented. ")
    else:
        sent_lens = np.array(
            [(len(s), len(t)) for s, t in zip(src, tgt)],
            dtype=np.dtype([("src", np.int64), ("tgt", np.int64)]),
        )
        if strategy == "src_tgt":
            sent_ind = argsort(sent_lens, order=("src", "tgt"))
        elif strategy == "tgt_src":
            sent_ind = argsort(sent_lens, order=("tgt", "src"))
        elif strategy == "src":
            sent_ind = argsort(sent_lens, order=("src"))
        elif strategy == "tgt":
            sent_ind = argsort(sent_lens, order=("tgt"))

        if long_first:
            sent_ind = sent_ind[::-1]

    src_padding_num = 0
    tgt_padding_num = 0
    max_seq_len = 0
    max_tgt_len = 0
    max_src_len = 0
    ind = 0
    batches = []
    batch = []

    while ind <= len(sent_lens):
        if (
            ind == len(sent_lens)
            or max(max_seq_len, max(sent_lens[sent_ind[ind]])) * (len(batch) + 1)
            > max_tokens
        ):
            if len(batch) == 0:
                ind += 1
                continue
            batches.append(batch)
            if ind == len(sent_lens):
                break
            batch = []
            max_seq_len = 0
            max_src_len = 0
            max_tgt_len = 0

        if sent_lens[sent_ind[ind]][0] > max_src_len:
            src_padding_num += (sent_lens[sent_ind[ind]][0] - max_src_len) * len(batch)
            max_src_len = sent_lens[sent_ind[ind]][0]
        else:
            src_padding_num += max_src_len - sent_lens[sent_ind[ind]][0]

        if sent_lens[sent_ind[ind]][1] > max_tgt_len:
            tgt_padding_num += (sent_lens[sent_ind[ind]][1] - max_tgt_len) * len(batch)
            max_tgt_len = sent_lens[sent_ind[ind]][1]
        else:
            tgt_padding_num += max_tgt_len - sent_lens[sent_ind[ind]][1]

        max_seq_len = max(max(sent_lens[sent_ind[ind]]), max_seq_len)

        batch.append(sent_ind[ind])

        ind += 1

    info["total_padding_num"] = src_padding_num + tgt_padding_num
    info["src_padding_num"] = src_padding_num
    info["tgt_padding_num"] = tgt_padding_num
    info["batch_num"] = len(batches)

    return batches, info


class LanguagePairDataset(Dataset):
    def __init__(self, src: MonolingualDataset, tgt: MonolingualDataset) -> None:
        super().__init__()
        assert len(src) == len(tgt)
        self.src = src
        self.tgt = tgt

    def __getitem__(self, index) -> T_co:
        return {
            "id": index,
            "data": {
                "src_lang": self.src[index]["data"]["source"],
                "tgt_lang": self.tgt[index]["data"]["source"],
                "target": self.tgt[index]["data"]["target"],
            },
        }

    def __len__(self):
        return len(self.src)


class LanguagePairIterableDataset(IterableDataset):
    def __init__(self, dataset: LanguagePairDataset, batch_sampler) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_sampler = batch_sampler

        self.padding()

    def padding(self):
        batch_sampler = []
        for batch in self.batch_sampler:
            sampler = batch
            batch = []
            max_tgt_len = 0
            max_src_len = 0
            for sample in sampler:
                data = self.dataset[sample]
                max_src_len = max(max_src_len, len(data["data"]["src_lang"]))
                max_tgt_len = max(max_tgt_len, len(data["data"]["tgt_lang"]))
            for sample in sampler:
                data = deepcopy(self.dataset[sample])
                data["data"]["src_lang"] = [
                    self.dataset.src.word_dict.padding_idx
                    for _ in range(max_src_len - len(data["data"]["src_lang"]))
                ] + data["data"]["src_lang"]
                data["data"]["tgt_lang"] += [
                    self.dataset.tgt.word_dict.padding_idx
                    for _ in range(max_tgt_len - len(data["data"]["tgt_lang"]))
                ]
                data["data"]["target"] += [
                    self.dataset.tgt.word_dict.padding_idx
                    for _ in range(max_tgt_len - len(data["data"]["target"]))
                ]
                batch.append(data)
            batch_sampler.append(batch)
        self.batch_sampler = batch_sampler

    def __iter__(self) -> Iterator[T_co]:
        batch_sampler = []
        for batch in self.batch_sampler:
            batch_sampler.append([batch[i] for i in RandomSampler(batch)])
        sample_ind = RandomSampler(batch_sampler)
        for ind in sample_ind:
            yield batch_sampler[ind]

    def __len__(self):
        return len(self.batch_sampler)


def prepare_language_pair_dataset(data_path, src_lang, tgt_lang, split):

    src = prepare_monolingual_dataset(data_path, src_lang, split, target="none")
    tgt = prepare_monolingual_dataset(data_path, tgt_lang, split, target="future")

    return LanguagePairDataset(src, tgt)


def prepare_dataloader(
    data_path,
    src_lang,
    tgt_lang,
    split,
    max_tokens,
    strategy="tgt_src",
    long_first=True,
):
    dataset = prepare_language_pair_dataset(data_path, src_lang, tgt_lang, split)
    batch_sampler, info = batch_by_size(
        dataset.src.data,
        dataset.tgt.data,
        max_tokens,
        strategy=strategy,
        long_first=long_first,
    )

    iter_dataset = LanguagePairIterableDataset(dataset, batch_sampler)

    return (
        DataLoader(iter_dataset, None, collate_fn=collate_fn, pin_memory=True),
        dataset.src.word_dict,
    )

