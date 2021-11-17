from argparse import ArgumentParser


def init_preprocess_options(parser: ArgumentParser):

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--src-lang", required=True)
    parser.add_argument("--tgt-lang", required=True)
    parser.add_argument("--dist-dir", default="data-bin")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vocab-name", default="bpevocab")
    parser.add_argument("--lines-per-thread", type=int, default=1000)
