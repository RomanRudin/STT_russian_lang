"""
data.py
=======
Подготовка данных для мультимодального восстановления пунктуации.

Делает три вещи:
  1. Загружает Google FLEURS (ru_ru) — пары (аудио, транскрипция с пунктуацией).
  2. Из транскрипции извлекает целевые метки трёх голов
     (пунктуация / абзац / капитализация) и «чистый» вход без пунктуации.
  3. Выравнивает слова по времени (forced alignment отдельной либой) и считает
     акустические признаки на каждое слово (паузы, длительность, F0, энергия).

Главный публичный объект — функция `build_examples(...)`, возвращающая список
`Example`, готовый к подаче в dataset.py.

Зависимости для полного режима (forced alignment + просодия):
    pip install datasets soundfile librosa
    pip install ctc-forced-aligner   # либо WhisperX / torchaudio MFA

Если эти либы/датасет недоступны (оффлайн, слабое железо), модуль
прозрачно переключается в text-only режим: метки берутся из текста, а
акустические признаки заполняются нулями (allow_text_only в DataConfig).
"""

from __future__ import annotations

import os
import re
import json
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np

from modules.config import (
    CHAR_TO_PUNCT,
    PUNCT2ID,
    PARA2ID,
    CAP2ID,
    ACOUSTIC_FEATURES,
    ACOUSTIC_NORM,
    ACOUSTIC_DIM,
    DataConfig,
)


# ---------------------------------------------------------------------------
# Структура одного обучающего примера
# ---------------------------------------------------------------------------
@dataclass
class Example:
    """Один пример = последовательность слов + три цепочки меток + акустика."""
    words: List[str]                 # слова БЕЗ пунктуации, в нижнем регистре
    punct_ids: List[int]             # знак после слова (PUNCT2ID)
    para_ids: List[int]              # абзац перед словом (PARA2ID)
    cap_ids: List[int]               # капитализация слова (CAP2ID)
    acoustic: np.ndarray             # (len(words), ACOUSTIC_DIM), уже нормализована
    has_acoustic: bool = True        # False -> признаки нулевые (text-only)
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.words)


# ---------------------------------------------------------------------------
# 1. Разбор транскрипции -> метки + чистый текст
# ---------------------------------------------------------------------------
# Знаки, которые мы умеем восстанавливать. «...» нормализуем в «…».
_PUNCT_CHARS = set(",.?!…")
_TOKEN_RE = re.compile(r"[^\s]+", re.UNICODE)


def _normalize_text(text: str) -> str:
    """Юникод-нормализация, унификация многоточий и кавычек, чистка пробелов."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("...", "…").replace("…", "…")
    text = text.replace("«", "").replace("»", "").replace('"', "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_trailing_punct(token: str) -> Tuple[str, str]:
    """
    Отделяет завершающий знак препинания от слова.
    'мир,'  -> ('мир', ',')   'да?!' -> ('да', '?')  (берём первый значимый знак)
    Возвращает (слово_без_знака, символ_знака | '').
    """
    punct = ""
    # снимаем хвостовые знаки; запоминаем «сильнейший» (предпочитаем . ? ! … над ,)
    stripped = token
    found = []
    while stripped and stripped[-1] in _PUNCT_CHARS:
        found.append(stripped[-1])
        stripped = stripped[:-1]
    if found:
        # приоритет: терминальные знаки важнее запятой
        for ch in ["…", "?", "!", ".", ","]:
            if ch in found:
                punct = ch
                break
    return stripped, punct


def _capitalization_label(raw_word: str) -> str:
    """Определяет метку капитализации по исходному (с регистром) слову."""
    letters = [c for c in raw_word if c.isalpha()]
    if not letters:
        return "LOWER"
    if all(c.isupper() for c in letters) and len(letters) > 1:
        return "UPPER"
    if letters[0].isupper():
        return "CAP"
    return "LOWER"


def parse_transcription(
    raw_transcription: str,
    paragraph_breaks: Optional[List[int]] = None,
) -> Tuple[List[str], List[int], List[int], List[int]]:
    """
    Превращает строку с пунктуацией в (words, punct_ids, para_ids, cap_ids).

    raw_transcription : текст с пунктуацией и регистром (как во FLEURS).
    paragraph_breaks  : индексы слов, ПЕРЕД которыми начинается новый абзац
                        (опционально; FLEURS почти не размечает абзацы —
                        см. примечание в build_examples).
    """
    text = _normalize_text(raw_transcription)
    raw_tokens = _TOKEN_RE.findall(text)

    words: List[str] = []
    punct_ids: List[int] = []
    cap_ids: List[int] = []

    for tok in raw_tokens:
        stripped, punct_char = _strip_trailing_punct(tok)
        # ведущие знаки (открывающая кавычка/тире) уже убраны в _normalize_text
        if not stripped:
            # токен был только из пунктуации — приклеиваем знак к предыдущему слову
            if punct_char and punct_ids:
                punct_ids[-1] = PUNCT2ID[CHAR_TO_PUNCT.get(punct_char, "O")]
            continue
        cap_ids.append(CAP2ID[_capitalization_label(stripped)])
        words.append(stripped.lower())
        punct_ids.append(PUNCT2ID[CHAR_TO_PUNCT.get(punct_char, "O")])

    # абзацы
    para_ids = [PARA2ID["NO_PARA"]] * len(words)
    if paragraph_breaks:
        for idx in paragraph_breaks:
            if 0 <= idx < len(words):
                para_ids[idx] = PARA2ID["PARA"]

    return words, punct_ids, para_ids, cap_ids


# ---------------------------------------------------------------------------
# 2. Forced alignment + акустические признаки
# ---------------------------------------------------------------------------
def _try_import_aligner():
    """Ленивая попытка импортнуть форсированный выравниватель."""
    try:
        import ctc_forced_aligner  # noqa: F401
        return "ctc_forced_aligner"
    except Exception:
        return None


def forced_align(
    audio: np.ndarray,
    sample_rate: int,
    words: List[str],
    cache_path: Optional[str] = None,
) -> Optional[List[Dict]]:
    """
    Возвращает список словарей {word, start, end} (сек) или None, если
    выравнивание недоступно. Результат кэшируется на диск (json).

    Реализация на ctc-forced-aligner. Для другого выравнивателя (WhisperX,
    MFA) достаточно переписать тело — формат выхода зафиксирован.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    backend = _try_import_aligner()
    if backend is None:
        return None

    try:
        import torch
        from ctc_forced_aligner import (
            load_alignment_model,
            generate_emissions,
            preprocess_text,
            get_alignments,
            get_spans,
            postprocess_results,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_alignment_model(device, dtype=torch.float32)

        wav = torch.from_numpy(audio).float()
        emissions, stride = generate_emissions(model, wav, device=device)
        tokens_starred, text_starred = preprocess_text(
            " ".join(words), romanize=True, language="rus"
        )
        segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
        word_ts = postprocess_results(text_starred, spans, stride, scores)

        result = [
            {"word": w["text"], "start": float(w["start"]), "end": float(w["end"])}
            for w in word_ts
        ]
    except Exception as e:  # выравнивание не удалось — мягко падаем в text-only
        print(f"[forced_align] не удалось выровнять ({e}); text-only для примера")
        return None

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    return result


def _f0_features(seg: np.ndarray, sr: int) -> Tuple[float, float]:
    """Медиана и наклон F0 на хвосте слова (последние ~150 мс)."""
    if seg.size < int(0.03 * sr):
        return ACOUSTIC_NORM["f0_end_median"]["mean"], 0.0
    try:
        import librosa
        tail = seg[-int(0.15 * sr):] if seg.size > int(0.15 * sr) else seg
        f0, _, _ = librosa.pyin(
            tail.astype(np.float32),
            fmin=70, fmax=400, sr=sr,
            frame_length=1024, hop_length=256,
        )
        f0 = f0[~np.isnan(f0)]
        if f0.size < 2:
            return ACOUSTIC_NORM["f0_end_median"]["mean"], 0.0
        median = float(np.median(f0))
        x = np.arange(f0.size)
        slope = float(np.polyfit(x, f0, 1)[0]) * f0.size  # суммарный рост по хвосту
        return median, slope
    except Exception:
        return ACOUSTIC_NORM["f0_end_median"]["mean"], 0.0


def compute_acoustic_features(
    audio: np.ndarray,
    sr: int,
    word_ts: List[Dict],
    words: List[str],
) -> np.ndarray:
    """
    Считает матрицу признаков (len(words), ACOUSTIC_DIM) по таймкодам слов.
    Если число выровненных слов != числу слов, выравнивает по минимуму, остаток
    заполняет нулями (надёжнее, чем падать).
    """
    n = len(words)
    feats = np.zeros((n, ACOUSTIC_DIM), dtype=np.float32)
    if not word_ts:
        return feats

    m = min(n, len(word_ts))
    for i in range(m):
        w = word_ts[i]
        start, end = float(w["start"]), float(w["end"])
        dur = max(end - start, 1e-3)

        pause_before = start - float(word_ts[i - 1]["end"]) if i > 0 else 0.0
        pause_after = (float(word_ts[i + 1]["start"]) - end) if i < len(word_ts) - 1 else 0.0
        n_chars = max(len(words[i]), 1)
        rate = dur / n_chars

        seg = audio[int(start * sr): int(end * sr)]
        f0_med, f0_slope = _f0_features(seg, sr)
        energy = float(np.sqrt(np.mean(seg[-int(0.1 * sr):] ** 2))) if seg.size else 0.0

        raw = {
            "pause_before": max(pause_before, 0.0),
            "pause_after": max(pause_after, 0.0),
            "word_duration": dur,
            "speech_rate": rate,
            "f0_end_median": f0_med,
            "f0_end_slope": f0_slope,
            "energy_end": energy,
        }
        feats[i] = _normalize_acoustic(raw)
    return feats


def _normalize_acoustic(raw: Dict[str, float]) -> np.ndarray:
    """Z-нормализация по ACOUSTIC_NORM, порядок строго по ACOUSTIC_FEATURES."""
    vec = np.zeros(ACOUSTIC_DIM, dtype=np.float32)
    for j, name in enumerate(ACOUSTIC_FEATURES):
        stat = ACOUSTIC_NORM[name]
        vec[j] = (raw.get(name, stat["mean"]) - stat["mean"]) / (stat["std"] + 1e-6)
    return vec


# ---------------------------------------------------------------------------
# 3. Сборка примеров из FLEURS
# ---------------------------------------------------------------------------
def load_fleurs(cfg: DataConfig, split: str = "train", limit: Optional[int] = None):
    """
    Загружает FLEURS (ru_ru). Возвращает HF Dataset либо None при недоступности.
    `limit` ограничивает число примеров (удобно для демо-прогона).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            cfg.dataset_name, cfg.lang, split=split, cache_dir=cfg.cache_dir,
            trust_remote_code=True,
        )
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds
    except Exception as e:
        print(f"[load_fleurs] датасет недоступен ({e}).")
        return None


def build_examples(
    cfg: DataConfig,
    split: str = "train",
    limit: Optional[int] = None,
    use_alignment: bool = True,
) -> List[Example]:
    """
    Главная функция модуля. Возвращает список Example для обучения/оценки.

    Логика:
      * грузим FLEURS;
      * на каждый сэмпл парсим транскрипцию -> метки + чистые слова;
      * при use_alignment делаем forced alignment и считаем акустику;
      * иначе (или при ошибке) — нулевые признаки (text-only).

    Про абзацы: FLEURS — это отдельные предложения без разметки абзацев,
    поэтому para_ids здесь почти всегда NO_PARA. Голова абзаца остаётся в
    модели рабочей; для реального обучения абзацам нужен корпус с абзацами
    (книги/субтитры) — это отдельный источник, легко добавить тем же Example.
    """
    ds = load_fleurs(cfg, split=split, limit=limit)
    examples: List[Example] = []

    if ds is None:
        if cfg.allow_text_only:
            print("[build_examples] FLEURS недоступен — возвращаю демо-примеры (text-only).")
            return _demo_examples()
        raise RuntimeError("FLEURS недоступен и text-only запрещён в конфиге.")

    aligner_ok = (_try_import_aligner() is not None) if use_alignment else False
    if use_alignment and not aligner_ok:
        print("[build_examples] forced-aligner не установлен — text-only режим.")

    for idx, sample in enumerate(ds):
        raw_tr = sample.get("raw_transcription") or sample.get("transcription") or ""
        words, punct_ids, para_ids, cap_ids = parse_transcription(raw_tr)
        if len(words) < 2:
            continue

        has_ac = False
        acoustic = np.zeros((len(words), ACOUSTIC_DIM), dtype=np.float32)

        if aligner_ok and "audio" in sample:
            audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"].get("sampling_rate", cfg.sample_rate)
            cache_path = os.path.join(cfg.alignment_dir, f"{split}_{idx}.json")
            word_ts = forced_align(audio, sr, words, cache_path=cache_path)
            if word_ts:
                acoustic = compute_acoustic_features(audio, sr, word_ts, words)
                has_ac = True

        examples.append(
            Example(
                words=words, punct_ids=punct_ids, para_ids=para_ids,
                cap_ids=cap_ids, acoustic=acoustic, has_acoustic=has_ac,
                meta={"id": sample.get("id", idx)},
            )
        )
    print(f"[build_examples] собрано {len(examples)} примеров "
          f"({'с акустикой' if aligner_ok else 'text-only'}).")
    return examples


def _demo_examples() -> List[Example]:
    """Несколько примеров «на сухую», чтобы код запускался без сети/датасета."""
    demo = [
        "Привет, как дела? Я давно тебя не видел!",
        "Сегодня хорошая погода. Может, прогуляемся по набережной…",
        "Что это было? Невероятно! Я не ожидал такого поворота событий.",
        "Москва — столица России, крупный экономический центр.",
    ]
    out = []
    for t in demo:
        words, p, par, cap = parse_transcription(t)
        out.append(Example(words, p, par, cap,
                           np.zeros((len(words), ACOUSTIC_DIM), np.float32),
                           has_acoustic=False))
    return out


# ---------------------------------------------------------------------------
# 4. Утилита: пересчёт статистик нормализации на train-акустике
# ---------------------------------------------------------------------------
def compute_acoustic_stats(examples: List[Example]) -> Dict[str, Dict[str, float]]:
    """
    Считает mean/std реальных (ещё ненормализованных) признаков по корпусу.
    Полезно, чтобы заменить грубые значения ACOUSTIC_NORM. Принимает примеры,
    созданные с raw-признаками (вариант для тонкой настройки — см. README).
    """
    arrs = [ex.acoustic for ex in examples if ex.has_acoustic]
    if not arrs:
        return ACOUSTIC_NORM
    stacked = np.concatenate(arrs, axis=0)
    stats = {}
    for j, name in enumerate(ACOUSTIC_FEATURES):
        col = stacked[:, j]
        stats[name] = {"mean": float(col.mean()), "std": float(col.std() + 1e-6)}
    return stats
