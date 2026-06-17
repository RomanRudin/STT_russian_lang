"""
models package
==============
Фабрика build_model(...) выбирает архитектуру по строковому имени, чтобы
ноутбук и train.py не зависели от конкретных классов.
"""

from __future__ import annotations

from typing import Optional

from ..config import LSTMConfig, TransformerConfig, PretrainedConfig
from .lstm_model import LSTMPunctuator
from .transformer_model import TransformerPunctuator
from .pretrained_model import PretrainedPunctuator, load_hf_tokenizer

__all__ = [
    "LSTMPunctuator", "TransformerPunctuator", "PretrainedPunctuator",
    "load_hf_tokenizer", "build_model",
]


def build_model(kind: str, vocab_size: Optional[int] = None,
                pad_id: int = 0, model_name: Optional[str] = None,
                use_acoustic: bool = True):
    """
    kind: 'lstm' | 'transformer' | 'pretrained'
    Для baseline-моделей нужен vocab_size; для предобученной — model_name.
    """
    kind = kind.lower()
    if kind == "lstm":
        cfg = LSTMConfig(use_acoustic=use_acoustic)
        if vocab_size:
            cfg.vocab_size = vocab_size
        return LSTMPunctuator(cfg, pad_id=pad_id)

    if kind == "transformer":
        cfg = TransformerConfig(use_acoustic=use_acoustic)
        if vocab_size:
            cfg.vocab_size = vocab_size
        return TransformerPunctuator(cfg, pad_id=pad_id)

    if kind == "pretrained":
        cfg = PretrainedConfig(use_acoustic=use_acoustic)
        if model_name:
            cfg.model_name = model_name
        return PretrainedPunctuator(cfg)

    raise ValueError(f"Неизвестный тип модели: {kind!r}")
