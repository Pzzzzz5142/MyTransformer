from argparse import ArgumentParser
import os.path as path
from multiprocessing import Pool
import torch
import os
import pickle


def init_options(parser: ArgumentParser):

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--source-lang", required=True)
    parser.add_argument("--dist-dir", default="data-bin")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vocab-name", default="bpevocab")
    parser.add_argument("--lines-per-thread", type=int, default=1000)


def handle_data_single_worker(src_lines, tgt_lines, word2ind):
    src_res = []
    tgt_res = []

    for src_line, tgt_line in zip(src_lines, tgt_lines):
        src_res.append(
            [
                word2ind[i] if i in word2ind else word2ind["<unk>"]
                for i in src_line.strip().split(" ")
            ]
        )
        tgt_res.append(
            [
                word2ind[i] if i in word2ind else word2ind["<unk>"]
                for i in tgt_line.strip().split(" ")
            ]
        )

    return src_res, tgt_res


def solve(args):

    word2ind = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    root_dir: str = args.data_path
    if root_dir[-1] == "/":
        root_dir = root_dir[:-1]
    dist_dir = path.join(args.dist_dir, path.basename(root_dir))

    with open(path.join(root_dir, args.vocab_name), "r") as vocab:
        vocab_size = 0
        for ind, line in enumerate(vocab.readlines()):
            line = line.strip().split()
            word2ind[line[0]] = ind + 4
            vocab_size = ind + 4

    os.makedirs(dist_dir)

    with open(path.join(dist_dir, "dict.txt"), "w") as fl:
        print(vocab_size, file=fl)
        for k, v in word2ind.items():
            print(f"{k} {v}", file=fl)

    for split in ["test", "train", "valid"]:
        pool = Pool(args.workers)
        with open(path.join(root_dir, f"{split}.{args.source_lang}"), "r") as src, open(
            path.join(root_dir, f"{split}.{args.target_lang}"), "r"
        ) as tgt:

            src_res = []
            tgt_res = []

            result = []

            src_lines = []
            tgt_lines = []
            for ind, (src_line, tgt_line) in enumerate(
                zip(src.readlines(), tgt.readlines())
            ):
                if ind > 0 and ind % args.lines_per_thread == 0:
                    result.append(
                        pool.apply_async(
                            handle_data_single_worker, (src_lines, tgt_lines, word2ind),
                        )
                    )
                    src_lines = []
                    tgt_lines = []

                src_lines.append(src_line)
                tgt_lines.append(tgt_line)

            if len(src_lines):
                result.append(
                    pool.apply_async(
                        handle_data_single_worker, (src_lines, tgt_lines, word2ind),
                    )
                )

            pool.close()
            pool.join()

            for res in result:
                res = res.get()
                src_res.append(res[0])
                tgt_res.append(res[1])

        with open(
            path.join(dist_dir, f"{split}.{args.source_lang}"), "wb"
        ) as src, open(path.join(dist_dir, f"{split}.{args.target_lang}"), "wb") as tgt:
            pickle.dump(src_res, src)
            pickle.dump(tgt_res, tgt)


if __name__ == "__main__":

    parser = ArgumentParser()

    init_options(parser)

    args = parser.parse_args()

    solve(args)
