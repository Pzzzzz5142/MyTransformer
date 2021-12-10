from argparse import ArgumentParser


def init_preprocess_options(parser: ArgumentParser):

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--src-lang", required=True)
    parser.add_argument("--tgt-lang", required=True)
    parser.add_argument("--dist-dir", default="data-bin")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vocab-name", default="bpevocab")
    parser.add_argument("--lines-per-thread", type=int, default=1000)


def init_train_options(parser: ArgumentParser):

    parser.add_argument("--seed", default=2, type=int)

    # training settings
    parser.add_argument("--adam-betas", default=(0.99, 0.999), type=tuple)
    parser.add_argument("--adam-eps", default=1e-8, type=float)
    parser.add_argument("--optim", choices=["adam", "adamw"], default="adam")
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Currently won't take effect. "
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing-eps", default=0.1, type=float)
    parser.add_argument("--model-config", default="config/base.yaml")
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", default=3, type=int)
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--update-freq", type=int, default=1)

    # data settings
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--src-lang", required=True)
    parser.add_argument("--tgt-lang", required=True)
    parser.add_argument("--batching-strategy", default="tgt_src")
    parser.add_argument("--batching-short-first", action="store_true", default=False)

    return parser
