"""
inference.py
============
Обёртка для инференса: единый интерфейс PunctuationRestorer для всех моделей.

restorer = PunctuationRestorer(model, backend="word",  vocab=vocab)
restorer = PunctuationRestorer(model, backend="rubert", tokenizer=tok)

text_out = restorer("привет как дела я давно тебя не видел")

Именно этот класс встраивается в SpeechToText-пайплайн как второе звено:
    звук -> [Whisper] -> текст без пунктуации -> [PunctuationRestorer] -> текст
"""

from __future__ import annotations

import re
from typing import List

import torch

from utils.labels import ID2PUNCT, ID2CASE, ID2PARA
from utils.preprocess import labels_to_text

_WORD_RE = re.compile(r"\w[\w\-]*", re.UNICODE)


class PunctuationRestorer:
    def __init__(
        self,
        model,
        backend: str,            # "word" (LSTM/Transformer) или "rubert"
        vocab=None,
        tokenizer=None,
        device: str | None = None,
        max_len: int = 128,
    ):
        assert backend in ("word", "rubert")
        self.model = model
        self.backend = backend
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w.lower() for w in _WORD_RE.findall(text)]

    @torch.no_grad()
    def __call__(self, text: str) -> str:
        words = self._tokenize(text)
        if not words:
            return ""
        words = words[: self.max_len]

        if self.backend == "word":
            punct, case, para = self._predict_word(words)
        else:
            punct, case, para = self._predict_rubert(words)

        return labels_to_text(words, punct, case, para)

    def _predict_word(self, words: List[str]):
        ids = torch.tensor([self.vocab.encode(words)], device=self.device)
        mask = torch.ones_like(ids)
        out = self.model(input_ids=ids, attention_mask=mask)
        p = out["punct"][0].argmax(-1).tolist()
        c = out["case"][0].argmax(-1).tolist()
        a = out["para"][0].argmax(-1).tolist()
        return (
            [ID2PUNCT[i] for i in p],
            [ID2CASE[i] for i in c],
            [ID2PARA[i] for i in a],
        )

    def _predict_rubert(self, words: List[str]):
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        word_ids = enc.word_ids()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        p_log = out["punct"][0].argmax(-1).tolist()
        c_log = out["case"][0].argmax(-1).tolist()
        a_log = out["para"][0].argmax(-1).tolist()

        # берём предсказание с ПЕРВОГО subword каждого слова
        punct = ["O"] * len(words)
        case = ["LOWER"] * len(words)
        para = ["NO_PARAGRAPH"] * len(words)
        prev = None
        for pos, wid in enumerate(word_ids):
            if wid is None or wid == prev:
                prev = wid
                continue
            punct[wid] = ID2PUNCT[p_log[pos]]
            case[wid] = ID2CASE[c_log[pos]]
            para[wid] = ID2PARA[a_log[pos]]
            prev = wid
        return punct, case, para
