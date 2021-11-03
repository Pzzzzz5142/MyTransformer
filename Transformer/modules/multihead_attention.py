import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num: int, model_dim: int):
        super().__init__()

        key_dim = model_dim // head_num

        assert key_dim * head_num == model_dim

        self.scale = 1 / math.sqrt(key_dim)

        self.head_num = head_num
        self.model_dim = model_dim
        self.key_dim = key_dim

        self.fc = nn.Linear(model_dim, model_dim)

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        net_input: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prev_tokens: Optional[torch.Tensor] = None,
        prev_token_mask: Optional[torch.Tensor] = None,
    ):
        """
            net_input: shape B x L x H
            encoder_padding_mask: shape B x L
            prev_tokens: Optional shape B x L x H
            pre_token_mask: Optional shape B x L
        """

        L = net_input.shape[0]

        net_input = net_input.transpose(0, 1)  # L x B x H

        q: torch.Tensor = self.q_proj(net_input)
        if prev_tokens == None:
            k = self.k_proj(net_input)
            v = self.v_proj(net_input)
        else:
            k = self.k_proj(prev_tokens)
            v = self.v_proj(prev_tokens)

        q = q.view(L, -1, self.key_dim)  # L x B*head_num x H//head_num
        k = k.view(L, -1, self.key_dim)
        v = v.view(L, -1, self.key_dim)
        q = q.transpose(0, 1)  # B*head_num x L x H/head_num
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q = q * self.scale
        k = k.transpose(1, 2)  # B*head_num  x H x L

        attn_weights = torch.bmm(q, k)  # B*head_num x L x L

        if attn_mask != None:
            attn_weights = attn_weights + attn_mask

        if encoder_padding_mask != None:
            encoder_padding_mask = encoder_padding_mask.unsqueeze(
                1
            ) * encoder_padding_mask.unsqueeze(2)
            attn_weights = attn_weights.masked_fill(encoder_padding_mask, float("-inf"))

        if prev_token_mask != None:
            prev_token_mask = prev_token_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(prev_token_mask, float("-inf"))

        attn_weights = attn_weights.softmax(-1)

        res = torch.bmm(attn_weights, v)

        res = res.transpose(0, 1)
        res = res.view(L, -1, self.model_dim)
        res = res.transpose(0, 1)

        res = self.fc(res)

        res=res+net_input
        res=F.layer_norm(res,res.shape[-2:])

        return res, attn_weights

