"""
src/acoustic/dataset.py

Loads and preprocesses two Russian speech datasets for Whisper fine-tuning:

  1. Google FLEURS  — loaded from HuggingFace (google/fleurs, ru_ru)
       ~10 h · clean read speech · no login required

  2. Golos          — loaded from LOCAL disk (downloaded manually)
       ~1240 h · crowd (~1000 h) + farfield (~240 h)
       Must be downloaded from https://sc.link/JpD (opus) or per-archive links.
       See DOWNLOAD_GUIDE.md for full instructions.

NOTE: SberDevices/Golos on HuggingFace is a stub repo with no Parquet data.
      Always use the direct download links and load from local disk.
"""

import os
import re
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_KEEP = re.compile(r"[^\u0400-\u04FF\u0030-\u0039\u0041-\u007Aa-z\s,\.!\?-]")


def normalise_text(text: str) -> str:
    """Lower-case, strip non-Russian chars, collapse whitespace."""
    text = text.lower().strip()
    text = _KEEP.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Column unification helper
# ---------------------------------------------------------------------------

_TRANSCRIPT_ALIASES = (
    "transcription", "text", "sentence", "transcript", "normalized_text", "label"
)


def _unify_column(ds: DatasetDict, source: str) -> DatasetDict:
    """Rename whatever transcript column exists → 'sentence'."""
    cols  = ds[next(iter(ds))].column_names
    if "sentence" in cols:
        return ds
    found = next((c for c in _TRANSCRIPT_ALIASES if c in cols), None)
    if found is None:
        raise ValueError(
            f"[{source}] No transcript column found. Available: {cols}\n"
            f"Expected one of: {_TRANSCRIPT_ALIASES}"
        )
    logger.info("[%s] Renaming column '%s' → 'sentence'", source, found)
    return ds.rename_column(found, "sentence")


def _keep_only(ds: DatasetDict, cols: tuple = ("audio", "sentence")) -> DatasetDict:
    to_drop = [c for c in ds[next(iter(ds))].column_names if c not in cols]
    return ds.remove_columns(to_drop) if to_drop else ds


# ---------------------------------------------------------------------------
# Audio augmentation
# ---------------------------------------------------------------------------

class AudioAugmenter:
    """Light on-the-fly augmentation for 16 kHz waveform tensors (1, T)."""

    def __init__(self, cfg: dict):
        self.noise_prob  = cfg.get("noise_prob", 0.3)
        self.noise_min   = cfg.get("noise_level_min", 0.001)
        self.noise_max   = cfg.get("noise_level_max", 0.015)
        self.speed_prob  = cfg.get("speed_prob", 0.2)
        self.speed_range = cfg.get("speed_range", [0.9, 1.1])

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() < self.noise_prob:
            wav = wav + torch.randn_like(wav) * random.uniform(
                self.noise_min, self.noise_max
            )
        if random.random() < self.speed_prob:
            factor = random.uniform(*self.speed_range)
            wav    = torchaudio.functional.resample(wav, int(16_000 * factor), 16_000)
        return torch.clamp(wav, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Per-sample preprocessing
# ---------------------------------------------------------------------------

def make_prepare_fn(
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    sample_rate: int = 16_000,
    augmenter: Optional[AudioAugmenter] = None,
):
    """
    Returns a datasets.map()-compatible preprocessing function.
    Input columns : 'audio' (Audio feature), 'sentence' (str)
    Output columns: 'input_features' (log-mel 80×3000), 'labels' (token ids)
    """

    def prepare(batch):
        array = np.array(batch["audio"]["array"], dtype=np.float32)

        if augmenter is not None:
            t     = torch.from_numpy(array).unsqueeze(0)
            array = augmenter(t).squeeze(0).numpy()

        feats = feature_extractor(
            array, sampling_rate=sample_rate, return_tensors="np"
        )
        batch["input_features"] = feats.input_features[0]
        batch["labels"]         = tokenizer(
            normalise_text(batch["sentence"])
        ).input_ids
        return batch

    return prepare


# ---------------------------------------------------------------------------
# FLEURS loader  (HuggingFace)
# ---------------------------------------------------------------------------

def load_fleurs(sample_rate: int = 16_000) -> DatasetDict:
    """
    Google FLEURS — Russian subset, loaded from HuggingFace.

    ~10 hours, studio-quality read speech, no login required.
    https://huggingface.co/datasets/google/fleurs
    """
    logger.info("Loading FLEURS (ru_ru) from HuggingFace …")
    ds = load_dataset(
        "google/fleurs",
        "ru_ru",
        trust_remote_code=True,   # required: FLEURS uses a custom loading script
    )
    ds = DatasetDict({k: ds[k] for k in ("train", "validation", "test") if k in ds})
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    ds = _unify_column(ds, "FLEURS")
    ds = _keep_only(ds)

    logger.info("FLEURS loaded: %s", {k: len(v) for k, v in ds.items()})
    return ds


# ---------------------------------------------------------------------------
# Golos loader  (local disk)
# ---------------------------------------------------------------------------

# TSV column names used in official Golos manifests.
# Format: <audio_path> TAB <duration> TAB <transcription>
# There is no header row in Golos TSV files.
_GOLOS_COL_PATH  = 0
_GOLOS_COL_DUR   = 1
_GOLOS_COL_TEXT  = 2


def _find_tsv_files(root: str, pattern: str) -> List[str]:
    """
    Recursively find TSV files under `root` whose name contains `pattern`.
    E.g. pattern='train' finds crowd_train.tsv, farfield_train.tsv, etc.
    """
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".tsv") and pattern in fname:
                matches.append(os.path.join(dirpath, fname))
    return sorted(matches)


def _resolve_audio_path(rel_path: str, root: str) -> Optional[str]:
    """
    Try several strategies to turn a relative path from a Golos TSV into
    an absolute path that exists on disk.

    Golos TSV paths look like:
        crowd/audio/0a1b2c3d.opus
        train_crowd0/0a1b2c3d.wav
        0a1b2c3d.opus
    """
    candidates = [
        rel_path,                              # absolute as-is
        os.path.join(root, rel_path),          # relative to golos root
        os.path.join(root, os.path.basename(rel_path)),  # just filename in root
    ]
    # Also try with a different extension (.opus ↔ .wav)
    for c in list(candidates):
        if c.endswith(".opus"):
            candidates.append(c[:-5] + ".wav")
        elif c.endswith(".wav"):
            candidates.append(c[:-4] + ".opus")

    return next((c for c in candidates if os.path.isfile(c)), None)


def _read_golos_tsv(
    tsv_path: str,
    root: str,
    max_dur: Optional[float] = 30.0,
    min_dur: Optional[float] = 0.5,
) -> List[Dict[str, str]]:
    """
    Read one Golos TSV manifest and return a list of {path, sentence} dicts.
    Rows with missing audio files or out-of-range duration are silently skipped.
    """
    records:         List[Dict[str, str]] = []
    skipped_missing: int = 0
    skipped_dur:     int = 0
    skipped_empty:   int = 0

    with open(tsv_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            rel_path = parts[_GOLOS_COL_PATH].strip()
            dur_str  = parts[_GOLOS_COL_DUR].strip()
            text     = parts[_GOLOS_COL_TEXT].strip()

            if not text:
                skipped_empty += 1
                continue

            # Fast duration filter — no audio decode needed
            try:
                dur = float(dur_str)
                if (max_dur and dur > max_dur) or (min_dur and dur < min_dur):
                    skipped_dur += 1
                    continue
            except ValueError:
                pass  # duration unparseable — keep the row

            abs_path = _resolve_audio_path(rel_path, root)
            if abs_path is None:
                skipped_missing += 1
                continue

            records.append({"path": abs_path, "sentence": text})

    logger.info(
        "  %s  kept=%d  skipped(missing=%d, dur=%d, empty=%d)",
        os.path.basename(tsv_path), len(records),
        skipped_missing, skipped_dur, skipped_empty,
    )
    return records


def load_golos(
    golos_path: str,
    subsets: List[str] = ("crowd", "farfield"),
    val_fraction: float = 0.02,
    sample_rate: int = 16_000,
    max_dur: float = 30.0,
    min_dur: float = 0.5,
) -> DatasetDict:
    """
    Load Golos from a locally extracted directory.

    Expected layout after extracting the downloaded archives
    (see DOWNLOAD_GUIDE.md for exact commands):

        <golos_path>/
            crowd_train.tsv          ← from train_crowd9.tar (contains all crowd manifests)
            farfield_train.tsv       ← from train_farfield.tar
            crowd_test.tsv           ← from test.tar
            farfield_test.tsv        ← from test.tar
            <audio files …>          ← from golos_opus.tar or train_crowd*.tar

    TSV format (tab-separated, NO header row):
        <audio_path>  <duration_seconds>  <transcription>

    Args:
        golos_path: Root directory of the extracted Golos dataset.
        subsets:    Which sub-corpora to include: 'crowd', 'farfield', or both.
        val_fraction: Fraction of train carved out as validation.
        sample_rate:  Target sample rate (16 000 Hz for Whisper).
        max_dur:      Skip clips longer than this (seconds).
        min_dur:      Skip clips shorter than this (seconds).
    """
    if not os.path.isdir(golos_path):
        raise FileNotFoundError(
            f"Golos directory not found: {golos_path}\n"
            "Download and extract from https://sc.link/JpD (opus) or\n"
            "the per-archive links at https://github.com/salute-developers/golos"
        )

    logger.info("Loading Golos from %s — subsets: %s", golos_path, list(subsets))

    train_records: List[Dict] = []
    test_records:  List[Dict] = []

    for subset in subsets:
        # ---- Train manifests ------------------------------------------------
        train_tsvs = _find_tsv_files(golos_path, f"{subset}_train")
        if not train_tsvs:
            # Some releases use a combined crowd_train.tsv for all crowd shards
            train_tsvs = _find_tsv_files(golos_path, "train")
            train_tsvs = [t for t in train_tsvs if subset in t]

        if not train_tsvs:
            logger.warning(
                "No train TSV found for subset '%s' in %s", subset, golos_path
            )
        for tsv in train_tsvs:
            train_records.extend(
                _read_golos_tsv(tsv, golos_path, max_dur=max_dur, min_dur=min_dur)
            )

        # ---- Test manifests -------------------------------------------------
        test_tsvs = _find_tsv_files(golos_path, f"{subset}_test")
        if not test_tsvs:
            test_tsvs = _find_tsv_files(golos_path, "test")
            test_tsvs = [t for t in test_tsvs if subset in t]

        for tsv in test_tsvs:
            test_records.extend(
                _read_golos_tsv(tsv, golos_path, max_dur=max_dur, min_dur=min_dur)
            )

    if not train_records:
        raise RuntimeError(
            f"No valid Golos training records found in {golos_path}.\n"
            "Check that TSV manifests exist and audio paths are resolvable.\n"
            "Run:  python scripts/preprocess.py --verify_only  to diagnose."
        )

    logger.info(
        "Golos raw records — train: %d  test: %d",
        len(train_records), len(test_records),
    )

    # Shuffle before carving validation so subset order is mixed
    random.shuffle(train_records)
    n_val          = max(1, int(len(train_records) * val_fraction))
    val_records    = train_records[:n_val]
    train_records  = train_records[n_val:]

    def _to_dataset(records: List[Dict]) -> Dataset:
        ds = Dataset.from_list(records)               # columns: path, sentence
        ds = ds.rename_column("path", "audio")
        ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
        return ds

    result = DatasetDict({
        "train":      _to_dataset(train_records),
        "validation": _to_dataset(val_records),
    })
    if test_records:
        result["test"] = _to_dataset(test_records)

    logger.info("Golos splits: %s", {k: len(v) for k, v in result.items()})
    return result


# ---------------------------------------------------------------------------
# Golos directory verifier  (called from preprocess.py --verify_only)
# ---------------------------------------------------------------------------

def verify_golos_layout(golos_path: str) -> None:
    """Print a diagnostic summary of the Golos directory layout."""
    if not os.path.isdir(golos_path):
        logger.error("Directory not found: %s", golos_path)
        return

    all_tsvs = _find_tsv_files(golos_path, "")   # find all TSVs
    logger.info("Golos directory: %s", golos_path)
    logger.info("TSV manifests found (%d):", len(all_tsvs))
    for tsv in all_tsvs:
        # Count lines quickly
        with open(tsv, encoding="utf-8") as f:
            n = sum(1 for l in f if l.strip())
        logger.info("  %-50s  %8d rows", os.path.relpath(tsv, golos_path), n)

    # Count audio files by extension
    counts: Dict[str, int] = {}
    for dirpath, _, filenames in os.walk(golos_path):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".opus", ".wav", ".mp3", ".flac"):
                counts[ext] = counts.get(ext, 0) + 1
    logger.info("Audio files: %s", counts)

    if not all_tsvs:
        logger.warning(
            "No TSV files found! Make sure you extracted train_crowd9.tar "
            "(which contains the manifests) and test.tar into %s", golos_path
        )


# ---------------------------------------------------------------------------
# Combined dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    cfg: dict,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
) -> DatasetDict:
    """
    Merges FLEURS and/or Golos, applies duration filtering, preprocessing
    and optional augmentation, and returns a DatasetDict with
    train / validation / test splits ready for Seq2SeqTrainer.

    Controlled by configs/acoustic.yaml → data section:
      use_fleurs:       true / false
      golos_path:       "/path/to/golos" or null
      golos_subsets:    ["crowd", "farfield"]
      golos_val_fraction: 0.02
    """
    sample_rate = cfg["data"]["sample_rate"]
    num_proc    = cfg["data"]["num_proc"]
    max_len     = cfg["data"]["max_audio_len_sec"]
    min_len     = cfg["data"]["min_audio_len_sec"]
    aug_cfg     = cfg.get("augmentation", {})
    data_cfg    = cfg["data"]

    train_parts, val_parts, test_parts = [], [], []

    # ---- FLEURS (HuggingFace) -----------------------------------------------
    if data_cfg.get("use_fleurs", True):
        fleurs = load_fleurs(sample_rate=sample_rate)
        train_parts.append(fleurs["train"])
        val_parts.append(fleurs["validation"])
        test_parts.append(fleurs["test"])

    # ---- Golos (local) -------------------------------------------------------
    golos_path = data_cfg.get("golos_path")
    if golos_path:
        golos = load_golos(
            golos_path=golos_path,
            subsets=data_cfg.get("golos_subsets", ["crowd", "farfield"]),
            val_fraction=data_cfg.get("golos_val_fraction", 0.02),
            sample_rate=sample_rate,
            max_dur=max_len,
            min_dur=min_len,
        )
        train_parts.append(golos["train"])
        val_parts.append(golos["validation"])
        if "test" in golos:
            test_parts.append(golos["test"])

    if not train_parts:
        raise ValueError(
            "No data sources configured!\n"
            "Set use_fleurs: true and/or golos_path: /path/to/golos "
            "in configs/acoustic.yaml."
        )

    merged = DatasetDict({
        "train":      concatenate_datasets(train_parts),
        "validation": concatenate_datasets(val_parts),
        "test":       concatenate_datasets(test_parts)
                      if test_parts else val_parts[0].select([]),
    })
    logger.info("Merged raw splits: %s", {k: len(v) for k, v in merged.items()})

    # ---- Fallback duration filter (catches anything the manifest filter missed)
    def duration_ok(ex):
        arr = ex["audio"]["array"]
        sr  = ex["audio"]["sampling_rate"]
        return min_len <= len(arr) / sr <= max_len

    merged = merged.filter(duration_ok, num_proc=num_proc)
    logger.info("After duration filter: %s", {k: len(v) for k, v in merged.items()})

    # ---- Feature extraction + augmentation ----------------------------------
    augmenter     = AudioAugmenter(aug_cfg) if aug_cfg.get("enabled") else None
    train_prepare = make_prepare_fn(feature_extractor, tokenizer, sample_rate, augmenter)
    eval_prepare  = make_prepare_fn(feature_extractor, tokenizer, sample_rate, None)

    merged["train"] = merged["train"].map(
        train_prepare,
        remove_columns=merged["train"].column_names,
        num_proc=num_proc,
        desc="Preprocessing train",
    )
    for split in ("validation", "test"):
        if len(merged[split]) > 0:
            merged[split] = merged[split].map(
                eval_prepare,
                remove_columns=merged[split].column_names,
                num_proc=num_proc,
                desc=f"Preprocessing {split}",
            )

    merged.set_format("torch")
    return merged


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads input_features within a batch and masks label padding with -100
    (so the loss function ignores padding positions).
    """
    processor: Any  # WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip leading BOS — the model re-adds it internally during forward()
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
