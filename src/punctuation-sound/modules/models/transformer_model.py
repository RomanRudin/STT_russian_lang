"""
models/transformer_model.py
===========================
Baseline-Transformer: энкодер на self-attention, обучаемый С НУЛЯ
(без предобучения), с late fusion акустики (вариант A).

Нужен как более сильная, чем LSTM, но всё ещё «своя» (непредобученная)
точка отсчёта — позволяет отделить вклад архитектуры от вклада предобучения.

Поток:
  слова -> Embedding + позиционные эмбеддинги -> N x TransformerEncoderLayer
        -> текстовое представление
  акустика -> AcousticEncoder
  concat -> MultiTaskHeads
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import TransformerConfig
from .heads import AcousticEncoder, MultiTaskHeads, fuse


class PositionalEncoding(nn.Module):
    """Классические синус/косинус позиционные эмбеддинги."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerPunctuator(nn.Module):
    def __init__(self, cfg: TransformerConfig, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=pad_id)
        self.pos = PositionalEncoding(cfg.embed_dim, cfg.max_len)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim, nhead=cfg.num_heads, dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

        if cfg.use_acoustic:
            self.acoustic_enc = AcousticEncoder(cfg.acoustic_dim, cfg.acoustic_hidden, cfg.dropout)
            fused_dim = cfg.embed_dim + self.acoustic_enc.out_dim
        else:
            self.acoustic_enc = None
            fused_dim = cfg.embed_dim

        self.heads = MultiTaskHeads(fused_dim, cfg.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                acoustic: Optional[torch.Tensor] = None, **_) -> Dict[str, torch.Tensor]:
        x = self.embed(input_ids) * math.sqrt(self.cfg.embed_dim)
        x = self.embed_dropout(self.pos(x))

        # TransformerEncoder ждёт True там, где НУЖНО маскировать (паддинг).
        pad_mask = ~attention_mask.bool()
        text_repr = self.encoder(x, src_key_padding_mask=pad_mask)

        if self.acoustic_enc is not None:
            ac = acoustic if acoustic is not None else torch.zeros(
                *input_ids.shape, self.cfg.acoustic_dim, device=input_ids.device)
            fused = fuse(text_repr, self.acoustic_enc(ac))
        else:
            fused = text_repr

        return self.heads(fused)
