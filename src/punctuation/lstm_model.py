"""
lstm_model.py
=============
Baseline #1 — двунаправленная LSTM поверх обучаемых эмбеддингов слов.

Три линейных классификатора-"головы" поверх скрытых состояний:
  punct (7 классов), case (3 класса), para (2 класса).

Это самая простая модель: эмбеддинги учатся с нуля, контекст ограничен.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.labels import NUM_PUNCT, NUM_CASE, NUM_PARA


class BiLSTMPunctuator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * 2  # bidirectional
        self.punct_head = nn.Linear(out_dim, NUM_PUNCT)
        self.case_head = nn.Linear(out_dim, NUM_CASE)
        self.para_head = nn.Linear(out_dim, NUM_PARA)

    def forward(self, input_ids, attention_mask=None, **_):
        emb = self.dropout(self.embedding(input_ids))
        h, _ = self.lstm(emb)
        h = self.dropout(h)
        return {
            "punct": self.punct_head(h),
            "case": self.case_head(h),
            "para": self.para_head(h),
        }


if __name__ == "__main__":
    m = BiLSTMPunctuator(vocab_size=200)
    x = torch.randint(0, 200, (2, 10))
    out = m(x)
    for k, v in out.items():
        print(k, tuple(v.shape))
