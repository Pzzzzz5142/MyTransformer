from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyWithLabelSmoothing(nn.Module):
    def __init__(self, vocab_size, eps):
        super().__init__()

        self.vocab_size = vocab_size
        self.eps = eps
        nn.CrossEntropyLoss()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        log_prob = net_output.view(-1, net_output.shape[-1]).log_softmax(-1)
        target = target.view(-1, target.shape[-1]).unsqueeze(-1)

        false_weight = self.eps / (self.vocab_size - 1)
        true_weight = 1 - self.eps - false_weight

        true_loss = -log_prob.gather(-1, target).squeeze(-1)
        false_loss = -log_prob.sum(-1)

        loss = true_loss * true_weight + false_loss * false_weight

        return loss

