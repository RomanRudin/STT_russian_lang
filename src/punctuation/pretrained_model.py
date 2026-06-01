"""
pretrained_model.py
====================
Предобученные модели: ruBERT с тремя головами token-classification.

Поддерживаются:
  "tiny"  -> cointegrated/rubert-tiny2     (лёгкая, быстрая, для демо/слабого GPU)
  "base"  -> DeepPavlov/rubert-base-cased  (тяжелее, точнее)

Особенность: ruBERT режет слова на subword-токены. Метку имеет ТОЛЬКО первый
subword каждого слова; остальные получают IGNORE_INDEX и не участвуют в loss.
Класс PretrainedDataset берёт на себя это выравнивание.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.labels import (
    NUM_PUNCT,
    NUM_CASE,
    NUM_PARA,
    PUNCT2ID,
    CASE2ID,
    PARA2ID,
    IGNORE_INDEX,
)
from utils.preprocess import text_to_labeled

MODEL_NAMES = {
    "tiny": "cointegrated/rubert-tiny2",
    "base": "DeepPavlov/rubert-base-cased",
}


class RuBertPunctuator(nn.Module):
    def __init__(self, variant: str = "tiny", dropout: float = 0.2):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        name = MODEL_NAMES[variant]
        self.config = AutoConfig.from_pretrained(name)
        self.encoder = AutoModel.from_pretrained(name)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.punct_head = nn.Linear(hidden, NUM_PUNCT)
        self.case_head = nn.Linear(hidden, NUM_CASE)
        self.para_head = nn.Linear(hidden, NUM_PARA)

    def forward(self, input_ids, attention_mask=None, **_):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)
        return {
            "punct": self.punct_head(h),
            "case": self.case_head(h),
            "para": self.para_head(h),
        }


def build_tokenizer(variant: str = "tiny"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAMES[variant])


class PretrainedDataset(Dataset):
    """
    Dataset для ruBERT: токенизирует слова через subword-токенизатор и
    выравнивает метки на первый subword каждого слова.
    """

    def __init__(self, texts: List[str], tokenizer, max_len: int = 128):
        self.examples = []
        for text in texts:
            words, punct, case, para = text_to_labeled(text)
            if not words:
                continue

            enc = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_len,
            )
            word_ids = enc.word_ids()

            p_lab, c_lab, a_lab = [], [], []
            prev = None
            for wid in word_ids:
                if wid is None:            # спецтокены [CLS]/[SEP]
                    p_lab.append(IGNORE_INDEX)
                    c_lab.append(IGNORE_INDEX)
                    a_lab.append(IGNORE_INDEX)
                elif wid != prev:          # первый subword слова -> ставим метку
                    p_lab.append(PUNCT2ID[punct[wid]])
                    c_lab.append(CASE2ID[case[wid]])
                    a_lab.append(PARA2ID[para[wid]])
                else:                      # продолжение слова -> игнор
                    p_lab.append(IGNORE_INDEX)
                    c_lab.append(IGNORE_INDEX)
                    a_lab.append(IGNORE_INDEX)
                prev = wid

            self.examples.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "word_ids": [w if w is not None else -1 for w in word_ids],
                    "punct": p_lab,
                    "case": c_lab,
                    "para": a_lab,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate(pad_token_id: int):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)

        def pad(seq, value):
            return seq + [value] * (max_len - len(seq))

        keys = ["input_ids", "attention_mask", "word_ids", "punct", "case", "para"]
        pads = {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "word_ids": -1,
            "punct": IGNORE_INDEX,
            "case": IGNORE_INDEX,
            "para": IGNORE_INDEX,
        }
        out = {}
        for k in keys:
            out[k] = torch.tensor([pad(b[k], pads[k]) for b in batch], dtype=torch.long)
        return out

    return collate


if __name__ == "__main__":
    # Лёгкая проверка без скачивания: только структура классов.
    print("Доступные варианты:", MODEL_NAMES)
