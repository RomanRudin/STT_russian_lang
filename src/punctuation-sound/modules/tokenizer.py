"""
tokenizer.py
============
Простой пословный токенизатор/словарь для baseline-моделей (LSTM, Transformer
с нуля). Предобученные модели (RuBERT) используют собственный subword-токенизатор
из HuggingFace и этот модуль не задействуют.

Словарь строится по обучающему корпусу (список Example). Поддерживает
спецтокены <pad>, <unk> и сериализацию в json.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import List, Dict

from .data import Example


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class WordVocab:
    """Отображение слово <-> id для пословных baseline-моделей."""

    def __init__(self, token2id: Dict[str, int]):
        self.token2id = token2id
        self.id2token = {i: t for t, i in token2id.items()}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, examples: List[Example], min_freq: int = 1,
              max_size: int = 50000) -> "WordVocab":
        counter: Counter = Counter()
        for ex in examples:
            counter.update(ex.words)

        token2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, freq in counter.most_common():
            if freq < min_freq:
                break
            if len(token2id) >= max_size:
                break
            token2id[token] = len(token2id)
        return cls(token2id)

    # --------------------------------------------------------------- encoding
    @property
    def pad_id(self) -> int:
        return self.token2id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token2id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, words: List[str]) -> List[int]:
        unk = self.unk_id
        return [self.token2id.get(w, unk) for w in words]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2token.get(i, UNK_TOKEN) for i in ids]

    # ----------------------------------------------------------------- io
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordVocab":
        with open(path, "r", encoding="utf-8") as f:
            token2id = json.load(f)
        return cls(token2id)
