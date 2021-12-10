import torch
from Transformer.data import WordDict
import numpy as np
import random

def handle_device(args):
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            UserWarning("No Cuda detected. Running on cpu.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def remove_bpe(sentence: str):
    sentence = sentence.split()
    ans = []
    tmp = ""
    for word in sentence:
        if word[-2:] == "@@":
            tmp += word[:-2]
        else:
            if tmp != "":
                ans.append(tmp + word)
                tmp = ""
            else:
                ans.append(word)

    return " ".join(ans)


def bpe_it(sentence: str, vocab_info: WordDict):
    sentence = sentence.strip().split()

    ans = []

    def solve_subword(word):
        if word in vocab_info.word_dict:
            return [word]
        for idx in range(len(word), -1, -1):
            if word[:idx] + "@@" in vocab_info.word_dict:
                sub_split = solve_subword(word[idx:])
                if sub_split != None:
                    return [word[:idx] + "@@"] + sub_split
        return None

    for word in sentence:
        split = solve_subword(word)
        if split != None:
            ans += split
        else:
            ans.append("<unk>")

    return " ".join(ans)

def ensure_reproducibility(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
