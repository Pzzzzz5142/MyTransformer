import torch
import torch.nn as nn


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        ...
