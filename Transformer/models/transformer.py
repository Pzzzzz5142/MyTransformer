import torch
import torch.nn as nn
from Transformer.modules import MultiHeadAttention, TransformerLayer
from typing import Optional
import math


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, ffn_dim, head_num, encoder_layers):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.extend(
            [
                TransformerLayer(head_num, model_dim, ffn_dim)
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        net_input: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):

        x = net_input

        for layer in self.layers:
            x, _ = layer(x, padding_mask, attn_mask)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, ffn_dim, head_num, decoder_layers):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.extend(
            [
                TransformerLayer(head_num, model_dim, ffn_dim)
                for _ in range(decoder_layers)
            ]
        )

    def forward(
        self,
        net_input: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prev_input: Optional[torch.Tensor] = None,
        prev_input_padding_mask: Optional[torch.Tensor] = None,
    ):

        x = net_input

        for layer in self.layers:
            x, _ = layer(
                x, padding_mask, attn_mask, prev_input, prev_input_padding_mask
            )

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        padding_idx,
        model_dim,
        ffn_dim,
        head_num,
        encoder_layers,
        decoder_layers,
        share_embeddings=True,
    ):
        super().__init__()

        self.padding_idx = padding_idx
        self.share_embeddings = share_embeddings
        self.model_dim = model_dim

        if share_embeddings:
            self.embedding = nn.Embedding(
                vocab_size, model_dim, padding_idx=self.padding_idx
            )
        else:
            self.encoder_emb = nn.Embedding(
                vocab_size[0], model_dim, padding_idx=self.padding_idx
            )
            self.decoder_emb = nn.Embedding(
                vocab_size[1], model_dim, padding_idx=self.padding_idx
            )

        self.encoder = TransformerEncoder(model_dim, ffn_dim, head_num, encoder_layers)
        self.decoder = TransformerDecoder(model_dim, ffn_dim, head_num, decoder_layers)

        self.fc = nn.Linear(
            model_dim, vocab_size if isinstance(vocab_size, int) else vocab_size[1]
        )

    def forward(self, input_tokens, output_tokens):

        en_padding_mask = input_tokens == self.padding_idx

        if self.share_embeddings:
            x = self.embedding(input_tokens)
        else:
            x = self.encoder_emb(input_tokens)
        x = x * math.sqrt(self.model_dim)

        pos = self.__generate_pos_matrix(x)
        x = x + pos  # add position embedding

        encoder_out = self.encoder(x, en_padding_mask)

        de_padding_mask = output_tokens == self.padding_idx

        if self.share_embeddings:
            x = self.embedding(output_tokens)
        else:
            x = self.decoder_emb(output_tokens)
        x = x * math.sqrt(self.model_dim)

        pos = self.__generate_pos_matrix(x)
        x = x + pos

        decoder_out = self.decoder(
            x,
            de_padding_mask,
            prev_input=encoder_out,
            prev_input_padding_mask=en_padding_mask,
        )

        predict = self.fc(decoder_out)

        return predict

    def __generate_pos_matrix(self, x: torch.Tensor):
        pos = [
            [
                math.cos(pos / (10000 ** (2 * i / self.model_dim)))
                if i & 1
                else math.sin(pos / (10000 ** (2 * i / self.model_dim)))
                for i in range(x.shape[-1])
            ]
            for pos in range(x.shape[-2])
        ]

        pos = x.new_tensor(pos)

        return pos

