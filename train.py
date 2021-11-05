from torch import nn
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from Transformer.models import Transformer
from Transformer.data import prepare_dataloader, PinMemoryBatch
from Transformer.criteration import CrossEntropyWithLabelSmoothing
from argparse import ArgumentParser
from torch.optim import AdamW
from Transformer.handle import TransformerLrScheduler
import yaml
from yaml import Loader
from tqdm import tqdm


def init_option(parser: ArgumentParser):

    # training settings
    parser.add_argument("--adam-betas", default=(0.99, 0.98), type=tuple)
    parser.add_argument("--adam-eps", default=1e-9, type=float)
    parser.add_argument("--label-smoothing-eps", default=0.1, type=float)
    parser.add_argument("--model-config", default="config/base.yaml")
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")

    # data settings
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--src-lang", required=True)
    parser.add_argument("--tgt-lang", required=True)
    parser.add_argument("--batching-strategy", default="tgt_src")
    parser.add_argument("--batching-long-first", type=bool, default=True)

    return parser


def train(
    model: nn.Module,
    criteration: nn.Module,
    train_data: DataLoader,
    optim: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
):
    for ind, samples in tqdm(enumerate(train_data)):
        samples = samples.to(device).get_batch()
        optim.zero_grad()
        loss, sample_size = criteration(model, **samples)
        loss.backward()
        optim.step()
        scheduler.step()

        if ind % 100 == 0:
            print(float(loss) / sample_size)


def valid():
    with torch.no_grad():
        ...


def trainer(args):

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            UserWarning("No Cuda detected. Running on cpu.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    valid_data, vocab_info = prepare_dataloader(
        args.data,
        args.src_lang,
        args.tgt_lang,
        "valid",
        args.max_tokens,
        args.batching_strategy,
        args.batching_long_first,
    )

    with open(args.model_config, "r", encoding="utf-8") as model_config:
        model_dict = yaml.load(model_config, Loader=Loader)
        model = Transformer(
            vocab_info.vocab_size, vocab_info.padding_idx, **model_dict
        ).to(device)

    train_data, _ = prepare_dataloader(
        args.data,
        args.src_lang,
        args.tgt_lang,
        "train",
        args.max_tokens,
        args.batching_strategy,
        args.batching_long_first,
    )

    print(model)

    optim = AdamW(model.parameters(), betas=args.adam_betas, eps=args.adam_eps)
    scheduler = TransformerLrScheduler(
        optim, model_dict["model_dim"], args.warmup_steps
    )
    criteration = CrossEntropyWithLabelSmoothing(args.label_smoothing_eps)

    train(model, criteration, train_data, optim, scheduler, device)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser = init_option(parser)

    args = parser.parse_args()

    trainer(args)
