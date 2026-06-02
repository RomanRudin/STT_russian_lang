"""
models/heads.py
===============
Переиспользуемые компоненты для всех трёх архитектур (вариант A, late fusion):

  * AcousticEncoder — небольшой MLP, сжимающий вектор просодических признаков
    слова (паузы / длительность / F0 / энергия) в скрытое представление.
  * MultiTaskHeads  — три линейных классификатора поверх объединённого
    (текст ⊕ акустика) представления: пунктуация / абзац / капитализация.

Так fusion одинаково устроен в LSTM, Transformer и RuBERT — отличается только
текстовый энкодер, что и есть смысл late fusion.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import NUM_PUNCT, NUM_PARA, NUM_CAP


class AcousticEncoder(nn.Module):
    """MLP-энкодер просодии: (B, T, acoustic_dim) -> (B, T, out_dim)."""

    def __init__(self, acoustic_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(acoustic_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out_dim = hidden

    def forward(self, acoustic: torch.Tensor) -> torch.Tensor:
        return self.net(acoustic)


class MultiTaskHeads(nn.Module):
    """
    Три головы token-classification поверх fused-представления.
    Возвращает словарь логитов: {"punct", "para", "cap"}.
    """

    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.punct = nn.Linear(in_dim, NUM_PUNCT)
        self.para = nn.Linear(in_dim, NUM_PARA)
        self.cap = nn.Linear(in_dim, NUM_CAP)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.dropout(x)
        return {"punct": self.punct(x), "para": self.para(x), "cap": self.cap(x)}


def fuse(text_repr: torch.Tensor,
         acoustic_repr: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Late fusion: конкатенация по последней оси. Если акустики нет
    (text-only инференс), возвращает только текстовое представление —
    но тогда головы должны быть инициализированы под text-only размерность.
    Для единообразия акустический энкодер всегда вызывается (на нулях), см.
    модели ниже, поэтому здесь просто конкатенация.
    """
    if acoustic_repr is None:
        return text_repr
    return torch.cat([text_repr, acoustic_repr], dim=-1)
