import torch


class WordDict(object):
    def __init__(self, word_dict_path) -> None:
        super().__init__()

        with open(word_dict_path, "r", encoding="utf-8") as f:
            self.vocab_size = int(f.readline().strip())
            self.word_dict = {}
            self.idx_dict = {}
            for line in f.readlines():
                word, ind = line.strip().split()
                ind = int(ind)
                self.word_dict[word] = ind
                self.idx_dict[ind] = word
                if word[0] == "<":
                    sp_tokens = word[1:-1]
                    if sp_tokens == "pad":
                        self.padding_idx = ind
                    elif sp_tokens == "unk":
                        self.unknown_idx = ind
                    elif sp_tokens == "bos":
                        self.bos_idx = ind
                    elif sp_tokens == "eos":
                        self.eos_idx = ind

    def idx2word(self, idx: int):
        return self.idx_dict[idx]

    def word2idx(self, word: str):
        return self.word_dict[word]

    def tokenize(self, sentence: str):
        sentence = [self.word_dict[word] for word in sentence.strip().split()]
        return sentence

    def detokenize(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            if len(tensor.shape) == 2:
                sentence = [
                    " ".join([self.idx_dict[idx] for idx in sent])
                    for sent in tensor.cpu().tolist()
                ]
            elif len(tensor.shape) == 1:
                sentence = " ".join(
                    [self.idx_dict[idx] for idx in tensor.cpu().tolist()]
                )
            else:
                raise Exception()
        else:
            sentence = " ".join([self.idx_dict[idx] for idx in tensor])

        return sentence
