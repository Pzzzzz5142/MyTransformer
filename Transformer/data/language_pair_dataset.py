import numpy as np
from numpy.core.fromnumeric import argsort, sort
from torch.utils.data import Dataset, BatchSampler
from torch.utils.data.dataset import T_co
from Transformer.data import prepare_monolingual_dataset, MonolingualDataset


def collate_fn(samples):  # form samples into batches
    ...


def batch_by_size(
    src, tgt, max_tokens, strategy="src_tgt", long_first=True
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
        return {"src_lang": self.src[index], "tgt_lang": self.tgt[index]}

    def __len__(self):
        return len(self.src)


def prepare_language_pair_dataset(data_path, src_lang, tgt_lang, split):

    src = prepare_monolingual_dataset(data_path, src_lang, split, target="present")
    tgt = prepare_monolingual_dataset(data_path, tgt_lang, split, target="future")

    return LanguagePairDataset(src, tgt)
