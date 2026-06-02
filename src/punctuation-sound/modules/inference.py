"""
inference.py
============
PunctuationRestorer — точка встраивания в SpeechToText-пайплайн.

Принимает текст БЕЗ пунктуации (и опционально пословные тайм-коды + аудио для
просодии), возвращает текст С пунктуацией, абзацами и капитализацией.

Два публичных метода:
  * restore(text)                         — text-only (быстрый путь).
  * restore_from_words(words, word_ts,    — мультимодальный: на вход слова и их
                       audio, sr)           таймкоды (как отдаёт Whisper), внутри
                                            считаются те же акустические признаки,
                                            что и при обучении.

Так модель пунктуации цепляется к выходу STT: Whisper отдаёт
[(слово, t0, t1), ...] -> restore_from_words(...) -> готовый текст.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import torch

from .config import (
    ID2PUNCT, ID2PARA, ID2CAP, PUNCT_TO_CHAR, ACOUSTIC_DIM,
)
from .data import compute_acoustic_features, _normalize_text, _TOKEN_RE


class PunctuationRestorer:
    """
    Универсальная обёртка над любой из трёх обученных моделей.

    Параметры
    ---------
    model        : обученная модель (LSTM / Transformer / Pretrained).
    kind         : 'lstm' | 'transformer' | 'pretrained' — определяет токенизацию.
    vocab        : WordVocab (для baseline) либо None.
    hf_tokenizer : HF-токенизатор (для pretrained) либо None.
    device       : 'cuda' / 'cpu'.
    """

    def __init__(self, model, kind: str, vocab=None, hf_tokenizer=None,
                 device: str = "cpu", max_len: int = 256):
        self.model = model.eval()
        self.kind = kind.lower()
        self.vocab = vocab
        self.tok = hf_tokenizer
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.max_len = max_len
        self.model.to(self.device)

    # ------------------------------------------------------------------ public
    def restore(self, text: str) -> str:
        """Text-only режим: акустика = нули."""
        words = self._split(text)
        if not words:
            return ""
        acoustic = np.zeros((len(words), ACOUSTIC_DIM), dtype=np.float32)
        labels = self._predict(words, acoustic)
        return self._assemble(words, labels)

    def restore_from_words(self, words: List[str],
                           word_ts: Optional[List[Dict]] = None,
                           audio: Optional[np.ndarray] = None,
                           sr: int = 16000) -> str:
        """
        Мультимодальный режим. words — токены из STT (без пунктуации);
        word_ts — [{word,start,end}] от Whisper/aligner; audio+sr — для F0/энергии.
        Если аудио не передано, считаются только временны́е признаки (паузы/длит.),
        остальные остаются нулевыми.
        """
        words = [w.lower() for w in words if w.strip()]
        if not words:
            return ""

        if word_ts:
            if audio is not None:
                acoustic = compute_acoustic_features(audio, sr, word_ts, words)
            else:
                acoustic = self._timing_only_features(words, word_ts)
        else:
            acoustic = np.zeros((len(words), ACOUSTIC_DIM), dtype=np.float32)

        labels = self._predict(words, acoustic)
        return self._assemble(words, labels)

    # --------------------------------------------------------------- internals
    @staticmethod
    def _split(text: str) -> List[str]:
        """Чистим возможную пунктуацию во входе и режем на слова (нижний регистр)."""
        text = _normalize_text(text)
        text = "".join(c for c in text if c not in ",.?!…")
        return [t.lower() for t in _TOKEN_RE.findall(text)]

    def _timing_only_features(self, words: List[str], word_ts: List[Dict]) -> np.ndarray:
        """Признаки без аудио: только паузы / длительность / темп (F0,energy=0)."""
        from .data import _normalize_acoustic
        n = len(words)
        feats = np.zeros((n, ACOUSTIC_DIM), dtype=np.float32)
        m = min(n, len(word_ts))
        for i in range(m):
            w = word_ts[i]
            start, end = float(w["start"]), float(w["end"])
            dur = max(end - start, 1e-3)
            pause_before = start - float(word_ts[i - 1]["end"]) if i > 0 else 0.0
            pause_after = (float(word_ts[i + 1]["start"]) - end) if i < len(word_ts) - 1 else 0.0
            raw = {
                "pause_before": max(pause_before, 0.0),
                "pause_after": max(pause_after, 0.0),
                "word_duration": dur,
                "speech_rate": dur / max(len(words[i]), 1),
                "f0_end_median": 160.0, "f0_end_slope": 0.0, "energy_end": 0.0,
            }
            feats[i] = _normalize_acoustic(raw)
        return feats

    @torch.no_grad()
    def _predict(self, words: List[str], acoustic: np.ndarray) -> Dict[str, List[int]]:
        """Прогоняет модель и возвращает по-словные id для трёх голов."""
        words = words[: self.max_len]
        acoustic = acoustic[: self.max_len]

        if self.kind in ("lstm", "transformer"):
            ids = torch.tensor([self.vocab.encode(words)], dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            ac = torch.tensor(acoustic[None], dtype=torch.float32, device=self.device)
            logits = self.model(input_ids=ids, attention_mask=mask, acoustic=ac)
            punct = logits["punct"][0].argmax(-1).cpu().tolist()
            para = logits["para"][0].argmax(-1).cpu().tolist()
            cap = logits["cap"][0].argmax(-1).cpu().tolist()
            return {"punct": punct, "para": para, "cap": cap}

        # pretrained: subword -> читаем решение с первого субтокена каждого слова
        enc = self.tok(words, is_split_into_words=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        word_ids = enc.word_ids()
        # акустика на субтокены
        ac_sub = np.zeros((enc["input_ids"].size(1), ACOUSTIC_DIM), dtype=np.float32)
        for j, wid in enumerate(word_ids):
            if wid is not None and wid < len(acoustic):
                ac_sub[j] = acoustic[wid]
        ac = torch.tensor(ac_sub[None], dtype=torch.float32, device=self.device)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"], acoustic=ac)

        p = logits["punct"][0].argmax(-1).cpu().tolist()
        pa = logits["para"][0].argmax(-1).cpu().tolist()
        c = logits["cap"][0].argmax(-1).cpu().tolist()

        punct, para, cap = [], [], []
        prev = None
        for j, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                punct.append(p[j]); para.append(pa[j]); cap.append(c[j])
            prev = wid
        return {"punct": punct, "para": para, "cap": cap}

    def _assemble(self, words: List[str], labels: Dict[str, List[int]]) -> str:
        """Собирает финальную строку: капитализация + знаки + абзацы."""
        punct, para, cap = labels["punct"], labels["para"], labels["cap"]
        out_tokens: List[str] = []
        for i, w in enumerate(words):
            # капитализация
            cap_lbl = ID2CAP.get(cap[i] if i < len(cap) else 0, "LOWER")
            if cap_lbl == "CAP":
                w = w[:1].upper() + w[1:]
            elif cap_lbl == "UPPER":
                w = w.upper()
            # знак после слова
            punct_lbl = ID2PUNCT.get(punct[i] if i < len(punct) else 0, "O")
            w = w + PUNCT_TO_CHAR.get(punct_lbl, "")
            # абзац перед словом
            para_lbl = ID2PARA.get(para[i] if i < len(para) else 0, "NO_PARA")
            sep = "\n\n" if (para_lbl == "PARA" and i > 0) else (" " if i > 0 else "")
            out_tokens.append(sep + w)

        text = "".join(out_tokens)
        # первая буква предложения с большой, если модель не пометила
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text


# ---------------------------------------------------------------------------
# Демонстрация склейки с STT (Whisper-заглушка)
# ---------------------------------------------------------------------------
class STTPunctuationPipeline:
    """
    Имитация полного пайплайна: audio -> STT(Whisper) -> текст без пунктуации
    -> PunctuationRestorer -> текст с пунктуацией.

    В реальном использовании `stt_fn` — это вызов вашей Whisper-модели,
    возвращающий dict {"words": [...], "word_timestamps": [{word,start,end}], "audio": np.ndarray}.
    Здесь принимается готовый словарь, чтобы модуль не тянул зависимость от Whisper.
    """

    def __init__(self, restorer: PunctuationRestorer):
        self.restorer = restorer

    def __call__(self, stt_output: Dict, sr: int = 16000) -> str:
        words = stt_output.get("words")
        word_ts = stt_output.get("word_timestamps")
        audio = stt_output.get("audio")
        if words is None and "text" in stt_output:
            return self.restorer.restore(stt_output["text"])
        return self.restorer.restore_from_words(words, word_ts=word_ts, audio=audio, sr=sr)
