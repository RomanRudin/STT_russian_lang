"""
models/lstm_model.py
====================
Baseline: двунаправленный LSTM с late fusion акустики (вариант A).

Поток:
  слова -> Embedding -> BiLSTM -> текстовое представление (B,T,2H)
  акустика -> AcousticEncoder -> (B,T,Ha)
  concat -> MultiTaskHeads -> {punct, para, cap}

Самая лёгкая модель, обучается с нуля, годится как нижняя планка качества.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..config import LSTMConfig
from .heads import AcousticEncoder, MultiTaskHeads, fuse


class LSTMPunctuator(nn.Module):
    def __init__(self, cfg: LSTMConfig, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            cfg.embed_dim, cfg.hidden_dim, num_layers=cfg.num_layers,
            batch_first=True, bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        text_dim = cfg.hidden_dim * 2

        if cfg.use_acoustic:
            self.acoustic_enc = AcousticEncoder(cfg.acoustic_dim, cfg.acoustic_hidden, cfg.dropout)
            fused_dim = text_dim + self.acoustic_enc.out_dim
        else:
            self.acoustic_enc = None
            fused_dim = text_dim

        self.heads = MultiTaskHeads(fused_dim, cfg.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                acoustic: Optional[torch.Tensor] = None, **_) -> Dict[str, torch.Tensor]:
        lengths = attention_mask.sum(dim=1).cpu()
        emb = self.embed(input_ids)

        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        text_repr, _ = pad_packed_sequence(out, batch_first=True,
                                           total_length=input_ids.size(1))

        if self.acoustic_enc is not None:
            ac = acoustic if acoustic is not None else torch.zeros(
                *input_ids.shape, self.cfg.acoustic_dim, device=input_ids.device)
            fused = fuse(text_repr, self.acoustic_enc(ac))
        else:
            fused = text_repr

        return self.heads(fused)
