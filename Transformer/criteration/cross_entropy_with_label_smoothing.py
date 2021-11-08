import torch
import torch.nn as nn


class CrossEntropyWithLabelSmoothing(nn.Module):
    def __init__(self, eps):
        super().__init__()

        self.eps = eps

    def forward(self, model, net_input: dict, target: torch.Tensor, *kargs, **kwargs):

        net_output: torch.Tensor = model(**net_input)

        padding_idx = model.vocab_info.padding_idx

        target = target.view(-1)
        log_prob = net_output.view(-1, net_output.shape[-1]).log_softmax(-1)[
            target != padding_idx
        ]
        target = target[target != padding_idx]

        false_weight = self.eps / (net_output.shape[-1] - 1)
        true_weight = 1 - self.eps - false_weight

        true_loss = -log_prob.gather(-1, target.unsqueeze(1)).squeeze(-1)
        false_loss = -log_prob.sum(-1)

        loss = true_loss * true_weight + false_loss * false_weight

        return loss.sum(), target.shape[0]

