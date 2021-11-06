from argparse import ArgumentParser
import torch
from Transformer.data import WordDict


def init_option(parser: ArgumentParser):
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dict-path", required=True)


def solve(args):
    ...


if __name__ == "__main__":
    parser = ArgumentParser()

    init_option(parser)

    args = parser.parse_args()

    solve(args)
