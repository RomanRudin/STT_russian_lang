"""
dataset.py
==========
PyTorch-обёртки над списком Example.

Два режима:
  * BaselineDataset  — для LSTM/Transformer: слова кодируются WordVocab,
                       акустика идёт как есть (1 вектор на слово).
  * PretrainedDataset — для RuBERT: subword-токенизация HF, метки и акустика
                       «размазываются» по субтокенам (метка на первом субтокене
                       слова, остальные = IGNORE_INDEX; акустика дублируется).

collate-функции делают паддинг до длины батча и строят attention_mask.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import IGNORE_INDEX, ACOUSTIC_DIM
from .data import Example
from .tokenizer import WordVocab


# ===========================================================================
# Baseline (пословный) датасет
# ===========================================================================
class BaselineDataset(Dataset):
    def __init__(self, examples: List[Example], vocab: WordVocab, max_len: int = 256):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        L = min(len(ex), self.max_len)
        return {
            "input_ids": torch.tensor(self.vocab.encode(ex.words[:L]), dtype=torch.long),
            "punct": torch.tensor(ex.punct_ids[:L], dtype=torch.long),
            "para": torch.tensor(ex.para_ids[:L], dtype=torch.long),
            "cap": torch.tensor(ex.cap_ids[:L], dtype=torch.long),
            "acoustic": torch.tensor(ex.acoustic[:L], dtype=torch.float32),
            "length": L,
        }


def baseline_collate(batch: List[Dict], pad_id: int = 0) -> Dict:
    """Паддинг батча пословных примеров до максимальной длины в батче."""
    maxlen = max(b["length"] for b in batch)
    B = len(batch)

    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    punct = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
    para = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
    cap = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
    acoustic = torch.zeros((B, maxlen, ACOUSTIC_DIM), dtype=torch.float32)
    mask = torch.zeros((B, maxlen), dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["length"]
        input_ids[i, :L] = b["input_ids"]
        punct[i, :L] = b["punct"]
        para[i, :L] = b["para"]
        cap[i, :L] = b["cap"]
        acoustic[i, :L] = b["acoustic"]
        mask[i, :L] = True

    return {
        "input_ids": input_ids, "punct": punct, "para": para, "cap": cap,
        "acoustic": acoustic, "attention_mask": mask,
    }


# ===========================================================================
# Предобученный (subword) датасет
# ===========================================================================
class PretrainedDataset(Dataset):
    """
    Размечает метки на уровне субтокенов. Метка слова ставится на ПЕРВЫЙ
    субтокен; продолжения помечаются IGNORE_INDEX, чтобы не штрафовать дважды.
    Акустический вектор слова копируется на все его субтокены (на инференсе
    решение читается с первого субтокена).
    """

    def __init__(self, examples: List[Example], hf_tokenizer, max_len: int = 256):
        self.examples = examples
        self.tok = hf_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        enc = self.tok(
            ex.words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
        )
        word_ids = enc.word_ids()

        punct, para, cap = [], [], []
        acoustic = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:                     # спецтокены [CLS]/[SEP]
                punct.append(IGNORE_INDEX); para.append(IGNORE_INDEX)
                cap.append(IGNORE_INDEX); acoustic.append(np.zeros(ACOUSTIC_DIM, np.float32))
            elif wid != prev_wid:               # первый субтокен слова
                punct.append(ex.punct_ids[wid]); para.append(ex.para_ids[wid])
                cap.append(ex.cap_ids[wid]); acoustic.append(ex.acoustic[wid])
            else:                               # продолжение слова
                punct.append(IGNORE_INDEX); para.append(IGNORE_INDEX)
                cap.append(IGNORE_INDEX); acoustic.append(ex.acoustic[wid])
            prev_wid = wid

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "punct": torch.tensor(punct, dtype=torch.long),
            "para": torch.tensor(para, dtype=torch.long),
            "cap": torch.tensor(cap, dtype=torch.long),
            "acoustic": torch.tensor(np.array(acoustic), dtype=torch.float32),
            "word_ids": [w if w is not None else -1 for w in word_ids],
            "length": len(enc["input_ids"]),
        }


def pretrained_collate(batch: List[Dict], pad_id: int = 0) -> Dict:
    maxlen = max(b["length"] for b in batch)
    B = len(batch)

    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attention = torch.zeros((B, maxlen), dtype=torch.long)
    punct = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
    para = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
    cap = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
    acoustic = torch.zeros((B, maxlen, ACOUSTIC_DIM), dtype=torch.float32)

    for i, b in enumerate(batch):
        L = b["length"]
        input_ids[i, :L] = b["input_ids"]
        attention[i, :L] = b["attention_mask"]
        punct[i, :L] = b["punct"]
        para[i, :L] = b["para"]
        cap[i, :L] = b["cap"]
        acoustic[i, :L] = b["acoustic"]

    return {
        "input_ids": input_ids, "attention_mask": attention,
        "punct": punct, "para": para, "cap": cap, "acoustic": acoustic,
    }
