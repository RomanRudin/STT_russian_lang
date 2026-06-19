"""
transformer_model.py
====================
Baseline #2 — Transformer-энкодер, обучаемый С НУЛЯ (без предобучения).

Обучаемые эмбеддинги слов + позиционные эмбеддинги + стек
nn.TransformerEncoderLayer. Те же три головы, что у LSTM.

Отличие от ruBERT: здесь НЕТ предобучения на больших корпусах — модель
видит только наш обучающий набор. Это честный baseline-трансформер.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.labels import NUM_PUNCT, NUM_CASE, NUM_PARA


class TransformerPunctuator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.2,
        max_len: int = 256,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.punct_head = nn.Linear(d_model, NUM_PUNCT)
        self.case_head = nn.Linear(d_model, NUM_CASE)
        self.para_head = nn.Linear(d_model, NUM_PARA)

    def forward(self, input_ids, attention_mask=None, **_):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(input_ids) + self.pos_emb(pos)
        h = self.dropout(h)

        # TransformerEncoder ждёт маску паддинга: True = игнорировать.
        pad_mask = None
        if attention_mask is not None:
            pad_mask = attention_mask == 0
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        return {
            "punct": self.punct_head(h),
            "case": self.case_head(h),
            "para": self.para_head(h),
        }


if __name__ == "__main__":
    m = TransformerPunctuator(vocab_size=200)
    x = torch.randint(0, 200, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    out = m(x, attention_mask=mask)
    for k, v in out.items():
        print(k, tuple(v.shape))
