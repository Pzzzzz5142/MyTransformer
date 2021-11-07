from argparse import ArgumentParser
import os
import torch
from Transformer.handle import handle_device
from Transformer.models import Transformer
import pickle


def init_option(parser: ArgumentParser):
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--src-lang", required=True)
    parser.add_argument("--tgt-lang", required=True)


def solve(args):
    device = handle_device(args)

    with open(args.model_path, "rb") as fl:
        model: Transformer = torch.load(fl, map_location=device)
        vocab_info = model.vocab_info

    with open(os.path.join(args.test_path, f"train.{args.src_lang}"), "rb") as src, open(
        os.path.join(args.test_path, f"train.{args.tgt_lang}"), "rb"
    ) as tgt:
        src_data = pickle.load(src)
        tgt_data = pickle.load(tgt)

    for src_sent, tgt_sent in zip(src_data, tgt_data):
        src_sent = vocab_info.detokenize(src_sent)
        tgt_sent = vocab_info.detokenize(tgt_sent)

        predict_sent = model.inference(src_sent,tgt_sent, device=device)

        print(f"Source:\n{src_sent}\nTarget:\n{tgt_sent}\nPredict:\n{predict_sent}\n")


if __name__ == "__main__":
    parser = ArgumentParser()

    init_option(parser)

    args = parser.parse_args()

    solve(args)
