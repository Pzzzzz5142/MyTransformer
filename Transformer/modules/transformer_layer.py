import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from Transformer.modules import MultiHeadAttention


class TransformerLayer(nn.Module):
    def __init__(self, head_num, model_dim, ffn_dim, drop_out=0.1, post_norm=True):
        super().__init__()

        self.drop_out = drop_out

        self.attn = MultiHeadAttention(head_num, model_dim)

        self.fc1 = nn.Linear(model_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, model_dim)

    def forward(
        self,
        net_input: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prev_input: Optional[torch.Tensor] = None,
        prev_input_padding_mask: Optional[torch.Tensor] = None,
    ):

        if prev_input != None:
            assert net_input.shape[0] == prev_input.shape[0]
            attn_mask = torch.triu(
                net_input.new_ones(
                    (net_input.shape[0], net_input.shape[1], net_input.shape[1]),
                    dtype=torch.bool,
                ),
                diagonal=1,
            )  # Future mask
            net_input, attn_weight = self.attn(net_input, padding_mask, attn_mask)

        mha_res, attn_weight = self.attn(
            net_input, padding_mask, None, prev_input, prev_input_padding_mask
        )

        x = self.fc1(mha_res)
        x = F.relu(x)
        x = self.fc2(x)

        x = F.dropout(x, self.drop_out)
        x = x + mha_res
        x = F.layer_norm(x, (x.shape[-1],))

        return x, attn_weight

