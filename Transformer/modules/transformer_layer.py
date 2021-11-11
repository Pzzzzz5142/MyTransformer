import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from Transformer.modules import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, head_num, model_dim, ffn_dim, drop_out=0.1, post_norm=True):
        super().__init__()

        self.drop_out = nn.Dropout(drop_out)
        self.layer_norm = nn.LayerNorm(model_dim)

        self.self_attn = MultiHeadAttention(head_num, model_dim)
        self.en_de_attn = MultiHeadAttention(head_num, model_dim)

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

        assert net_input.shape[0] == prev_input.shape[0]
        attn_mask = torch.triu(
            net_input.new_ones(
                (net_input.shape[0], net_input.shape[1], net_input.shape[1]),
                dtype=torch.bool,
            ),
            diagonal=1,
        )  # Future mask
        x, attn_weight = self.self_attn(net_input, padding_mask, attn_mask)

        x, attn_weight = self.en_de_attn(
            x, padding_mask, None, prev_input, prev_input_padding_mask
        )

        res = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = self.drop_out(x)
        x = x + res
        x = self.layer_norm(x)

        return x, attn_weight


class TransformerEncoderLayer(nn.Module):
    def __init__(self, head_num, model_dim, ffn_dim, drop_out=0.1, post_norm=True):
        super().__init__()

        self.drop_out = nn.Dropout(drop_out)
        self.layer_norm = nn.LayerNorm(model_dim)

        self.self_attn = MultiHeadAttention(head_num, model_dim)

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
        x, attn_weight = self.self_attn(net_input, padding_mask, attn_mask)

        res = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = self.drop_out(x)
        x = x + res
        x = self.layer_norm(x)

        return x, attn_weight
