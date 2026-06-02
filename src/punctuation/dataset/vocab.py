"""
vocab.py
========
Словарь слов и torch.Dataset для baseline-моделей (BiLSTM, baseline-Transformer),
которые работают на уровне СЛОВ (в отличие от ruBERT — там subword-токенизатор).

WordVocab     — отображение слово <-> id со спец-токенами <pad>/<unk>.
PunctDataset  — превращает список "сырых" текстов в тензоры:
                input_ids + три набора меток (punct/case/para).
collate_fn    — паддинг батча до максимальной длины.
"""

from __future__ import annotations

import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from utils.labels import PUNCT2ID, CASE2ID, PARA2ID, IGNORE_INDEX
from utils.preprocess import text_to_labeled

PAD, UNK = "<pad>", "<unk>"


class WordVocab:
    def __init__(self, token2id: Dict[str, int] | None = None):
        if token2id is None:
            token2id = {PAD: 0, UNK: 1}
        self.token2id = token2id
        self.id2token = {i: t for t, i in token2id.items()}

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token2id[UNK]

    def __len__(self) -> int:
        return len(self.token2id)

    @classmethod
    def build(cls, texts: List[str], min_freq: int = 1) -> "WordVocab":
        from collections import Counter

        counter: Counter = Counter()
        for t in texts:
            tokens, *_ = text_to_labeled(t)
            counter.update(tokens)
        token2id = {PAD: 0, UNK: 1}
        for tok, freq in counter.most_common():
            if freq >= min_freq:
                token2id[tok] = len(token2id)
        return cls(token2id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2id.get(t, self.unk_id) for t in tokens]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordVocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


class PunctDataset(Dataset):
    """Dataset для word-level моделей: один пример = один текст."""

    def __init__(self, texts: List[str], vocab: WordVocab, max_len: int = 128):
        self.examples = []
        self.vocab = vocab
        self.max_len = max_len
        for text in texts:
            tokens, punct, case, para = text_to_labeled(text)
            if not tokens:
                continue
            tokens = tokens[:max_len]
            punct = punct[:max_len]
            case = case[:max_len]
            para = para[:max_len]
            self.examples.append(
                {
                    "input_ids": vocab.encode(tokens),
                    "punct": [PUNCT2ID[l] for l in punct],
                    "case": [CASE2ID[l] for l in case],
                    "para": [PARA2ID[l] for l in para],
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def collate_fn(batch, pad_id: int = 0):
    """Паддинг батча. Метки паддятся IGNORE_INDEX, чтобы не учитывать в loss."""
    max_len = max(len(b["input_ids"]) for b in batch)

    def pad(seq, value):
        return seq + [value] * (max_len - len(seq))

    input_ids, punct, case, para, mask = [], [], [], [], []
    for b in batch:
        n = len(b["input_ids"])
        input_ids.append(pad(b["input_ids"], pad_id))
        punct.append(pad(b["punct"], IGNORE_INDEX))
        case.append(pad(b["case"], IGNORE_INDEX))
        para.append(pad(b["para"], IGNORE_INDEX))
        mask.append([1] * n + [0] * (max_len - n))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "punct": torch.tensor(punct, dtype=torch.long),
        "case": torch.tensor(case, dtype=torch.long),
        "para": torch.tensor(para, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


if __name__ == "__main__":
    from dataset.dataset import load_texts

    texts = load_texts("synthetic")
    v = WordVocab.build(texts)
    print("Размер словаря:", len(v))
    ds = PunctDataset(texts, v)
    print("Примеров:", len(ds))
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dl))
    for k, val in batch.items():
        print(f"  {k}: {tuple(val.shape)}")
