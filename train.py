import os
from argparse import ArgumentParser

import torch
import yaml
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from yaml import Loader

from Transformer.criteration import CrossEntropyWithLabelSmoothing
from Transformer.data import prepare_dataloader
from Transformer.handle import (
    TransformerLrScheduler,
    handle_device,
    ensure_reproducibility,
    init_train_options,
)
from Transformer.models import Transformer


def train(
    epoch: int,
    update_freq: int,
    model: nn.Module,
    criteration: nn.Module,
    train_data: DataLoader,
    valid_data: DataLoader,
    optim: Optimizer,
    scheduler: _LRScheduler,
    save_dir: str,
    device: torch.device,
):
    total_loss = 0
    total_sample = 0
    model.train()
    optim.zero_grad()
    ppl = 0
    training_epoch = 0
    for ind, samples in enumerate(tqdm(train_data)):  # Training
        samples = samples.to(device).get_batch()
        ind = ind + 1
        loss, logging_info = criteration(model, **samples)
        sample_size = logging_info["valid tokens num"]
        ppl += logging_info["ppl"]
        training_epoch += 1
        loss.backward()
        if ind % update_freq == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad()
        total_loss += float(loss)
        total_sample += int(sample_size)

        if (ind // update_freq) % 100 == 0 and ind % update_freq == 0:
            print(
                f"Epoch: {epoch} Training loss: {float(total_loss) / total_sample} ppl: {ppl/training_epoch} lr: {float(optim.param_groups[0]['lr'])}"
            )
            total_loss = 0
            total_sample = 0
            ppl = 0
            training_epoch = 0

    with torch.no_grad():  # Validating
        total_loss = 0
        total_sample = 0
        model.eval()
        for samples in tqdm(valid_data):
            samples = samples.to(device).get_batch()
            loss, logging_info = criteration(model, **samples)
            sample_size = logging_info["valid tokens num"]
            ppl += logging_info["ppl"]
            training_epoch += 1
            total_loss += loss
            total_sample += sample_size
        print(
            f"Epoch: {epoch} Valid loss: {float(total_loss / total_sample)} ppl: {ppl/training_epoch}"
        )

    with open(os.path.join(save_dir, f"epoch{epoch}.pt"), "wb") as fl:
        torch.save(model, fl)


def trainer(args):

    device = handle_device(args)
    ensure_reproducibility(args.seed)

    save_dir = args.save_dir.strip()

    if save_dir == "":
        save_dir = "checkpoint"

    os.makedirs(save_dir, exist_ok=True)

    valid_data, vocab_info = prepare_dataloader(
        args.data,
        args.src_lang,
        args.tgt_lang,
        "valid",
        args.max_tokens,
        args.batching_strategy,
        not args.batching_short_first,
    )

    with open(args.model_config, "r", encoding="utf-8") as model_config:
        model_dict = yaml.load(model_config, Loader=Loader)
        model = Transformer(vocab_info, **model_dict).to(device)
    train_data, _ = prepare_dataloader(
        args.data,
        args.src_lang,
        args.tgt_lang,
        "train",
        args.max_tokens,
        args.batching_strategy,
        not args.batching_short_first,
    )

    print(model)

    if args.optim == "adam":
        optim = Adam(
            model.parameters(),
            lr=args.lr,
            betas=args.adam_betas,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    else:
        optim = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=args.adam_betas,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    scheduler = TransformerLrScheduler(
        optim, model_dict["model_dim"], args.warmup_steps
    )
    criteration = CrossEntropyWithLabelSmoothing(args.label_smoothing_eps)

    for epoch in range(args.epoch):
        train(
            epoch,
            args.update_freq,
            model,
            criteration,
            train_data,
            valid_data,
            optim,
            scheduler,
            save_dir,
            device,
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser = init_train_options(parser)

    args = parser.parse_args()

    trainer(args)
