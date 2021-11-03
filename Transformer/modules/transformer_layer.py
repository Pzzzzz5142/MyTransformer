import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from Transformer.modules import MultiHeadAttention


class TansformerLayer(nn.Module):
    def __init__(self, head_num, model_dim, ffn_dim, post_norm=True):
        super().__init__()

        self.attn = MultiHeadAttention(head_num, model_dim)

        self.layer_norm = nn.LayerNorm()

        self.fc1 = nn.Linear(model_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, model_dim)

    def forward(
        self,
        net_input: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prev_tokens: Optional[torch.Tensor] = None,
        prev_token_mask: Optional[torch.Tensor] = None,
    ):

        if prev_tokens != None:
            attn_mask = torch.triu(
                net_input.new_ones(net_input.shape, dtype=torch.bool), diagonal=1,
            )  # Future mask
        else:
            attn_mask = None

        mha_res, attn_weight = self.attn(
            net_input, encoder_padding_mask, attn_mask, prev_tokens, prev_token_mask
        )

        x = self.fc1(mha_res)
        x = F.relu(x)
        x = self.fc2(x)

        x = x + mha_res
        x = F.layer_norm(x, x.shape[-2:])

        return x, attn_weight

