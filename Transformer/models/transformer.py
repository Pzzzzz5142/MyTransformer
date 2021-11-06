import torch
import torch.nn as nn
from Transformer.modules import TransformerLayer
from Transformer.data import WordDict
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
        vocab_info: WordDict,
        model_dim,
        ffn_dim,
        head_num,
        encoder_layers,
        decoder_layers,
        share_embeddings=True,
    ):
        super().__init__()

        self.padding_idx = vocab_info.padding_idx
        self.share_embeddings = share_embeddings
        self.model_dim = model_dim
        self.vocab_info = vocab_info

        vocab_size = vocab_info.vocab_size

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

    def forward(self, input_tokens, output_tokens) -> torch.Tensor:

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

    @torch.no_grad()
    def inference(self, source, beam_size=5):
        source = self.vocab_info.tokenize([source])

        source = torch.tensor(source)  # 1 x L

        net_output = torch.tensor([[self.vocab_info.bos_idx]])  # 1 x 1

        predict = self.forward(source, net_output)  #  1 x 1 x vocab_size
        predict = -predict.log_softmax(-1)

        predict_prob, predict_tokens = predict.topk(
            beam_size, -1, largest=False
        )  # 1 x 1 x beam_size

        net_output = torch.cat(
            [net_output.expand((beam_size, 1), predict_tokens.view(beam_size, 1))],
            dim=-1,
        )  # beam_size x 2
        total_prob = predict_prob.view(-1)

        while True:
            predict = self.forward(source, net_output)[:, -1, :].reshape(beam_size, -1)
            predict = -predict.log_softmax(-1)  # beam_size x vocab_size

            predict_prob, predict_tokens = predict.topk(
                beam_size, -1, largest=False
            )  # beam_size x beam_size

            net_output = (
                net_output.unsqueeze(1)
                .expand((beam_size, beam_size, net_output.shape[-1]))
                .reshape(beam_size * beam_size, -1)
            )  # beam_size*beam_size x L
            total_prob = (
                total_prob.unsqueeze(1).expand((beam_size, beam_size)).reshape(-1)
            )  # beam_size*beam_size
            predict_tokens = predict_tokens.view(beam_size * beam_size, 1)
            predict_prob = predict_prob.view(beam_size * beam_size, 1)

            net_output = torch.cat(
                [net_output, predict_tokens], dim=-1
            )  # beam_size*beam_size x L+1
            total_prob = total_prob + predict_prob  # beam_size*beam_size

            _, net_output_topk = total_prob.topk(
                beam_size, dim=-1, largest=False
            )  # beam_size

            net_output = net_output.index_select(0, net_output_topk)  # beam_size x L+1
            sentences = self.vocab_info.detokenize(net_output)
            print("\n".join(sentences))

            for sentence in sentences:
                last_token = sentence.split()[-1]
                if last_token == self.vocab_info.word2idx(self.vocab_info.eos_idx):
                    return sentence

