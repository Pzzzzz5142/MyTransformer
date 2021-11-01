import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self,head_num):
        super().__init__()

    def forward(self,net_input):
        ...