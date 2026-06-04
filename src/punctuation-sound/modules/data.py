"""
data.py
=======
Подготовка данных для мультимодального восстановления пунктуации.

Делает три вещи:
  1. Загружает корпус русской начитанной речи. ОСНОВНОЙ корпус — M-AILABS
     (русские аудиокниги: LibriVox/Gutenberg), где текст несёт ПОЛНУЮ книжную
     пунктуацию (запятые, точки, ? ! …) и реальные абзацы. FLEURS оставлен как
     опциональная добавка для разнообразия дикторов/тематики.
  2. Из текста извлекает целевые метки трёх голов (пунктуация / абзац /
     капитализация) и «чистый» вход без пунктуации.
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

from .config import (
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


def parse_document(raw_text: str) -> Tuple[List[str], List[int], List[int], List[int]]:
    """
    Разбор МНОГОАБЗАЦНОГО текста (книжный фрагмент M-AILABS).

    В отличие от parse_transcription, дополнительно извлекает границы абзацев:
    разрыв абзаца — это перевод строки (одинарный \\n или пустая строка между
    блоками). Слово, начинающее новый абзац, помечается PARA. Так голова `para`
    получает реальный обучающий сигнал (которого нет во FLEURS).

    M-AILABS обычно отдаёт по одному предложению на сэмпл (абзацев нет), но если
    несколько предложений склеены в один фрагмент с переносами — мы их используем.
    """
    # нормализуем разные переводы строк; пустые строки и одиночные \n считаем границей
    text = unicodedata.normalize("NFC", raw_text.replace("\r\n", "\n").replace("\r", "\n"))
    paragraphs = [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]

    words: List[str] = []
    punct_ids: List[int] = []
    para_ids: List[int] = []
    cap_ids: List[int] = []

    for pi, para in enumerate(paragraphs):
        pw, pp, _, pc = parse_transcription(para)
        if not pw:
            continue
        # первое слово абзаца (кроме самого первого в документе) -> PARA
        start_idx = len(words)
        words.extend(pw); punct_ids.extend(pp); cap_ids.extend(pc)
        para_block = [PARA2ID["NO_PARA"]] * len(pw)
        if pi > 0 and start_idx > 0:
            para_block[0] = PARA2ID["PARA"]
        para_ids.extend(para_block)

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
# 3. Загрузка корпусов (M-AILABS основной, FLEURS — опциональное смешивание)
# ---------------------------------------------------------------------------
def _pick_text_field(sample: Dict, candidates: List[str]) -> str:
    """Возвращает значение первого присутствующего текстового поля."""
    for name in candidates:
        val = sample.get(name)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def load_mailabs_from_archive(cfg: DataConfig, limit: Optional[int] = None):
    """
    Загружает русский M-AILABS напрямую из официального архива caito.de
    (НЕ через HuggingFace Hub — работает, когда Hub недоступен).

    Качает ru_RU.tgz (~оф. источник, формат LJSpeech: дерево папок с wavs/ и
    metadata.csv вида `id|оригинальный_текст|нормализованный_текст`), распаковывает
    в cfg.cache_dir и собирает список dict-сэмплов {sentence, audio_path}.

    Возвращает список сэмплов (не HF Dataset) либо None. Аудио читается лениво
    при вычислении акустики; для text-only режима аудио не нужно.
    """
    import urllib.request
    import tarfile
    import csv as _csv

    url = cfg.mailabs_archive_url
    cache = os.path.abspath(cfg.cache_dir)
    os.makedirs(cache, exist_ok=True)
    tgz_path = os.path.join(cache, "ru_RU.tgz")
    extract_dir = os.path.join(cache, "ru_RU")

    # 1) скачать архив (если ещё нет)
    if not os.path.exists(extract_dir):
        if not os.path.exists(tgz_path):
            try:
                print(f"[mailabs-archive] скачиваю {url} ...")
                print("  (~5-6 ГБ, это разовая операция; затем всё из кэша)")
                urllib.request.urlretrieve(url, tgz_path)
            except Exception as e:
                print(f"[mailabs-archive] не удалось скачать архив ({e}).")
                print("  Скачайте ru_RU.tgz вручную и положите в:", tgz_path)
                return None
        # 2) распаковать
        try:
            print("[mailabs-archive] распаковываю архив ...")
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(cache)
        except Exception as e:
            print(f"[mailabs-archive] ошибка распаковки ({e}).")
            return None

    # 3) собрать все metadata.csv (LJSpeech-формат) рекурсивно
    samples = []
    for root, _dirs, files in os.walk(cache):
        if "metadata.csv" in files:
            meta = os.path.join(root, "metadata.csv")
            wav_dir = os.path.join(root, "wavs")
            try:
                with open(meta, encoding="utf-8") as f:
                    for row in _csv.reader(f, delimiter="|"):
                        if len(row) < 2:
                            continue
                        clip_id = row[0]
                        # столбец 1 — оригинальный текст с пунктуацией (нужен нам);
                        # столбец 2 (если есть) — нормализованный (без пунктуации)
                        sentence = row[1]
                        wav = os.path.join(wav_dir, clip_id + ".wav")
                        samples.append({"sentence": sentence, "audio_path": wav})
            except Exception:
                continue

    if not samples:
        print("[mailabs-archive] не найдено metadata.csv после распаковки.")
        return None

    print(f"[mailabs-archive] собрано {len(samples)} клипов из архива.")
    if limit:
        samples = samples[:limit]
    return samples


def load_mailabs(cfg: DataConfig, split: str = "train", limit: Optional[int] = None):
    """
    Загружает M-AILABS (русские аудиокниги). Порядок попыток:
      1) официальный архив caito.de (надёжно, без HF Hub) — если cfg.use_archive;
      2) HuggingFace Hub по списку имён + автопоиск.

    M-AILABS идёт одним набором — train/val/test нарезаются сами (по seed).
    Возвращает (samples, is_hf): список сэмплов и флаг, HF это Dataset или
    список dict из архива (формат сэмпла одинаков по ключам sentence/audio_path|audio).
    """
    # --- маршрут 1: официальный архив (не зависит от HF Hub) ---
    if getattr(cfg, "use_archive", True):
        arch = load_mailabs_from_archive(cfg, limit=None)
        if arch is not None:
            return _split_samples(arch, cfg, split, limit), False

    # --- маршрут 2: HuggingFace Hub ---
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[load_mailabs] библиотека datasets недоступна ({e}).")
        return None, False

    base = None
    last_err = None
    candidates = list(cfg.mailabs_repos)

    # автопоиск по Hub API, если хотим и список кандидатов можно дополнить
    if cfg.mailabs_autosearch:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            found = api.list_datasets(search="m-ailabs", limit=50)
            ru_like = []
            for d in found:
                did = getattr(d, "id", "")
                low = did.lower()
                if "ailabs" in low and ("ru" in low or "rus" in low):
                    ru_like.append(did)
            for did in ru_like:
                if did not in candidates:
                    candidates.insert(0, did)
            if ru_like:
                print(f"[load_mailabs] автопоиск нашёл кандидатов: {ru_like}")
        except Exception as e:
            print(f"[load_mailabs] автопоиск недоступен ({e}).")

    for repo in candidates:
        try:
            base = load_dataset(repo, split="train", cache_dir=cfg.cache_dir,
                                trust_remote_code=True)
            print(f"[load_mailabs] загружен репозиторий: {repo} ({len(base)} клипов).")
            break
        except Exception as e:
            last_err = e
            continue
    if base is None:
        print(f"[load_mailabs] не удалось загрузить M-AILABS. Проверенные имена: "
              f"{candidates}. Последняя ошибка: {last_err}.")
        print("  Совет: оставьте cfg.data.use_archive=True для загрузки с caito.de,")
        print("  либо скачайте ru_RU.tgz вручную в cfg.data.cache_dir.")
        return None, False

    return _split_samples(base, cfg, split, limit, is_hf=True), True


def _split_samples(base, cfg: DataConfig, split: str,
                   limit: Optional[int], is_hf: bool = False):
    """Детерминированный train/val/test split для HF Dataset или списка dict."""
    n = len(base)
    n_test = int(n * cfg.test_ratio)
    n_val = int(n * cfg.val_ratio)
    bounds = {
        "test": (0, n_test),
        "validation": (n_test, n_test + n_val),
        "train": (n_test + n_val, n),
    }
    lo, hi = bounds.get(split, bounds["train"])

    if is_hf:
        base = base.shuffle(seed=42)
        ds = base.select(range(lo, hi))
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds
    else:
        import random as _r
        idx = list(range(n))
        _r.Random(42).shuffle(idx)
        chosen = idx[lo:hi]
        if limit:
            chosen = chosen[:limit]
        return [base[i] for i in chosen]


def examples_from_pkl(pkl_path: str, cfg: DataConfig,
                      use_alignment: bool = True,
                      limit: Optional[int] = None) -> List[Example]:
    """
    Загружает готовые Example из .pkl (созданного prepare_mailabs.py) и при
    наличии forced-aligner ДОСЧИТЫВАЕТ акустику по meta['audio_path'].

    Это самый надёжный путь, когда HuggingFace Hub недоступен: данные готовятся
    автономным скриптом из архива caito.de, а ноутбук просто их подхватывает.
    """
    import pickle
    with open(pkl_path, "rb") as f:
        examples: List[Example] = pickle.load(f)
    if limit:
        examples = examples[:limit]

    aligner_ok = (_try_import_aligner() is not None) if use_alignment else False
    if not aligner_ok:
        if use_alignment:
            print("[examples_from_pkl] forced-aligner не установлен — text-only.")
        print(f"[examples_from_pkl] загружено {len(examples)} примеров (text-only).")
        return examples

    # досчитываем акустику
    import soundfile as sf
    done = 0
    for i, ex in enumerate(examples):
        wav = ex.meta.get("audio_path")
        if not wav or not os.path.exists(wav):
            continue
        try:
            audio, sr = sf.read(wav, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            cache_path = os.path.join(cfg.alignment_dir, f"pkl_{i}.json")
            word_ts = forced_align(audio, sr, ex.words, cache_path=cache_path)
            if word_ts:
                ex.acoustic = compute_acoustic_features(audio, sr, word_ts, ex.words)
                ex.has_acoustic = True
                done += 1
        except Exception:
            continue
    print(f"[examples_from_pkl] загружено {len(examples)}, акустика посчитана для {done}.")
    return examples


def load_fleurs(cfg: DataConfig, split: str = "train", limit: Optional[int] = None):
    """
    Загружает FLEURS (ru_ru). Возвращает HF Dataset либо None при недоступности.
    Оставлен для опционального СМЕШИВАНИЯ с M-AILABS (разнообразие дикторов/тем).
    FLEURS почти не несёт ? ! … и абзацев — основной корпус теперь M-AILABS.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "google/fleurs", cfg.lang, split=split, cache_dir=cfg.cache_dir,
            trust_remote_code=True,
        )
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds
    except Exception as e:
        print(f"[load_fleurs] датасет недоступен ({e}).")
        return None


def _sample_to_example(sample: Dict, idx: int, cfg: DataConfig,
                       split: str, aligner_ok: bool, text_field: List[str],
                       use_documents: bool) -> Optional[Example]:
    """Превращает один HF-сэмпл в Example (с акустикой, если возможно)."""
    raw = _pick_text_field(sample, text_field)
    if not raw:
        return None

    # M-AILABS — книжный текст: пытаемся извлечь абзацы (use_documents=True);
    # FLEURS — одно предложение: обычный parse_transcription.
    if use_documents:
        words, punct_ids, para_ids, cap_ids = parse_document(raw)
    else:
        words, punct_ids, para_ids, cap_ids = parse_transcription(raw)
    if len(words) < cfg.min_words:
        return None

    has_ac = False
    acoustic = np.zeros((len(words), ACOUSTIC_DIM), dtype=np.float32)
    if aligner_ok:
        audio, sr = None, cfg.sample_rate
        # вариант 1: HF-поле audio с массивом
        if "audio" in sample and sample["audio"] is not None and isinstance(sample["audio"], dict):
            audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"].get("sampling_rate", cfg.sample_rate)
        # вариант 2: путь к wav (архив caito.de)
        elif sample.get("audio_path") and os.path.exists(sample["audio_path"]):
            try:
                import soundfile as sf
                audio, sr = sf.read(sample["audio_path"], dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)  # стерео -> моно
            except Exception:
                audio = None
        if audio is not None:
            cache_path = os.path.join(cfg.alignment_dir, f"{split}_{idx}.json")
            word_ts = forced_align(audio, sr, words, cache_path=cache_path)
            if word_ts:
                acoustic = compute_acoustic_features(audio, sr, word_ts, words)
                has_ac = True

    return Example(
        words=words, punct_ids=punct_ids, para_ids=para_ids,
        cap_ids=cap_ids, acoustic=acoustic, has_acoustic=has_ac,
        meta={"id": sample.get("id", idx)},
    )


def build_examples(
    cfg: DataConfig,
    split: str = "train",
    limit: Optional[int] = None,
    use_alignment: bool = True,
    source: str = "mailabs",
    mix_fleurs: bool = False,
) -> List[Example]:
    """
    Главная функция модуля. Возвращает список Example для обучения/оценки.

    Параметры
    ---------
    source       : основной корпус — "mailabs" (по умолчанию) или "fleurs".
    mix_fleurs   : если True и source="mailabs", добавляет FLEURS для разнообразия.
    use_alignment: при True и установленном forced-aligner считает акустику
                   (паузы/F0/энергия); иначе — text-only (нулевые признаки).

    M-AILABS (аудиокниги) несёт полную пунктуацию (? ! …) и реальные абзацы,
    поэтому классы QUESTION/EXCLAM/ELLIPSIS и голова PARA получают сигнал —
    в отличие от FLEURS.
    """
    aligner_ok = (_try_import_aligner() is not None) if use_alignment else False
    if use_alignment and not aligner_ok:
        print("[build_examples] forced-aligner не установлен — text-only режим.")

    examples: List[Example] = []

    # --- основной корпус ---
    if source == "mailabs":
        ds, _is_hf = load_mailabs(cfg, split=split, limit=limit)
        if ds is not None:
            for idx, sample in enumerate(ds):
                ex = _sample_to_example(sample, idx, cfg, split, aligner_ok,
                                        cfg.text_field_candidates, use_documents=True)
                if ex:
                    examples.append(ex)
    elif source == "fleurs":
        ds = load_fleurs(cfg, split=split, limit=limit)
        if ds is not None:
            for idx, sample in enumerate(ds):
                ex = _sample_to_example(sample, idx, cfg, split, aligner_ok,
                                        ["raw_transcription", "transcription"],
                                        use_documents=False)
                if ex:
                    examples.append(ex)

    # --- опциональное смешивание с FLEURS ---
    if source == "mailabs" and mix_fleurs:
        ds_f = load_fleurs(cfg, split=split if split != "validation" else "validation",
                           limit=limit)
        if ds_f is not None:
            for idx, sample in enumerate(ds_f):
                ex = _sample_to_example(sample, 10_000_000 + idx, cfg, split, aligner_ok,
                                        ["raw_transcription", "transcription"],
                                        use_documents=False)
                if ex:
                    examples.append(ex)

    # --- fallback: ничего не загрузилось ---
    if not examples:
        if cfg.allow_text_only:
            print("[build_examples] корпус недоступен — возвращаю демо-примеры (text-only).")
            return _demo_examples()
        raise RuntimeError("Корпус недоступен и text-only запрещён в конфиге.")

    print(f"[build_examples] собрано {len(examples)} примеров из '{source}'"
          f"{' + fleurs' if (source=='mailabs' and mix_fleurs) else ''} "
          f"({'с акустикой' if aligner_ok else 'text-only'}).")
    return examples


def _demo_examples(n_repeat: int = 60) -> List[Example]:
    """
    Запасные примеры, чтобы код исполнялся без сети/датасета.

    ВАЖНО: это НЕ настоящий корпус. Если build_examples вернул их — значит
    M-AILABS не загрузился, и любые метрики на них бессмысленны (фактически
    обучение на горстке предложений). Размножаем до n_repeat, чтобы хотя бы
    не падал цикл обучения, и громко предупреждаем.
    """
    print("=" * 70)
    print("ВНИМАНИЕ: используются ДЕМО-примеры (M-AILABS не загрузился).")
    print("Это НЕ реальный корпус — метрики на них не имеют смысла.")
    print("Укажите рабочее имя датасета в cfg.data.mailabs_repos и перезапустите.")
    print("=" * 70)
    demo_docs = [
        "Привет, как дела? Я давно тебя не видел!\nСегодня хорошая погода. "
        "Может, прогуляемся по набережной…",
        "Что это было? Невероятно! Я не ожидал такого поворота событий.\n"
        "Москва — столица России, крупный экономический центр.",
        "Он медленно открыл дверь. За ней никого не было… "
        "Куда же все подевались?\nСтранно, очень странно.",
        "Книга лежала на столе, раскрытая на середине. "
        "Кто её читал? И зачем оставил здесь?",
        "Мы долго шли по лесу, усталые и голодные. "
        "Наконец показалась деревня!\nЛюди встретили нас радушно.",
    ]
    base = []
    for t in demo_docs:
        words, p, par, cap = parse_document(t)
        base.append(Example(words, p, par, cap,
                            np.zeros((len(words), ACOUSTIC_DIM), np.float32),
                            has_acoustic=False))
    return base * n_repeat


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
