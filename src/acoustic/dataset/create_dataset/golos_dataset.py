import os
import random
import logging
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict, Audio

logger = logging.getLogger(__name__)

# TSV column indices
_PATH_COL = 0
_DUR_COL = 1
_TEXT_COL = 2

def _find_tsv_files(root: str, pattern: str) -> List[str]:
    """Find TSV files containing `pattern` in filename under root."""
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".tsv") and pattern in fname:
                matches.append(os.path.join(dirpath, fname))
    return sorted(matches)

def _resolve_audio_path(rel_path: str, root: str) -> Optional[str]:
    """Resolve relative audio path to absolute existing file."""
    candidates = [
        rel_path,
        os.path.join(root, rel_path),
        os.path.join(root, os.path.basename(rel_path)),
    ]
    # Try different extensions
    for c in list(candidates):
        if c.endswith(".opus"):
            candidates.append(c[:-5] + ".wav")
        elif c.endswith(".wav"):
            candidates.append(c[:-4] + ".opus")
    return next((c for c in candidates if os.path.isfile(c)), None)

def _read_golos_tsv(tsv_path: str, root: str, max_dur: float, min_dur: float) -> List[Dict[str, str]]:
    """Read one TSV manifest and return list of {'path': ..., 'sentence': ...}."""
    records = []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            rel_path = parts[_PATH_COL].strip()
            dur_str = parts[_DUR_COL].strip()
            text = parts[_TEXT_COL].strip()
            if not text:
                continue
            try:
                dur = float(dur_str)
                if dur > max_dur or dur < min_dur:
                    continue
            except ValueError:
                pass
            abs_path = _resolve_audio_path(rel_path, root)
            if abs_path is None:
                continue
            records.append({"path": abs_path, "sentence": text})
    return records

def create_golos(cfg: Dict[str, Any]) -> DatasetDict:
    """
    Load Golos dataset from local directory.

    Args:
        cfg: Configuration with dataset.sources.golos.path, subsets, val_fraction,
             and dataset.params for duration limits.

    Returns:
        DatasetDict with 'train', 'validation', and optionally 'test'.
    """
    golos_cfg = cfg['dataset']['sources']['golos']
    path = golos_cfg.get('path')
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(f"Golos path not found: {path}")

    subsets = golos_cfg.get('subsets', ['crowd', 'farfield'])
    val_frac = golos_cfg.get('val_fraction', 0.02)
    max_dur = cfg['dataset']['params']['max_duration']
    min_dur = cfg['dataset']['params']['min_duration']
    sample_rate = cfg['dataset']['params']['sample_rate']

    logger.info("Loading Golos from %s, subsets: %s", path, subsets)

    train_records = []
    test_records = []

    for subset in subsets:
        # Train manifests
        train_tsvs = _find_tsv_files(path, f"{subset}_train")
        if not train_tsvs:
            train_tsvs = _find_tsv_files(path, "train")
            train_tsvs = [t for t in train_tsvs if subset in t]
        for tsv in train_tsvs:
            train_records.extend(_read_golos_tsv(tsv, path, max_dur, min_dur))

        # Test manifests
        test_tsvs = _find_tsv_files(path, f"{subset}_test")
        if not test_tsvs:
            test_tsvs = _find_tsv_files(path, "test")
            test_tsvs = [t for t in test_tsvs if subset in t]
        for tsv in test_tsvs:
            test_records.extend(_read_golos_tsv(tsv, path, max_dur, min_dur))

    if not train_records:
        raise RuntimeError(f"No valid training records found in {path}")

    random.shuffle(train_records)
    n_val = max(1, int(len(train_records) * val_frac))
    val_records = train_records[:n_val]
    train_records = train_records[n_val:]

    def to_dataset(records):
        ds = Dataset.from_list(records)
        ds = ds.rename_column("path", "audio")
        ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
        return ds

    result = DatasetDict({
        "train": to_dataset(train_records),
        "validation": to_dataset(val_records),
    })
    if test_records:
        result["test"] = to_dataset(test_records)

    logger.info("Golos splits: %s", {k: len(v) for k, v in result.items()})
    return result