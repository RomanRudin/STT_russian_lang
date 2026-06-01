import re
import unicodedata
from datasets import DatasetDict
from typing import Any, Dict
from tqdm import tqdm

# Keep Cyrillic, digits, basic punctuation and spaces
_KEEP = re.compile(r"[^\u0400-\u04FF\u0030-\u0039\u0041-\u007Aa-z\s,\.!\?-]")

def normalise_text(text: str) -> str:
    """Lower-case, strip non-Russian chars, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = _KEEP.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def apply(dataset: DatasetDict, cfg: Dict[str, Any]) -> DatasetDict:
    """
    Normalize text in the 'sentence' column.

    Args:
        dataset: DatasetDict with 'sentence' column.
        cfg: Unused, kept for interface.

    Returns:
        DatasetDict with normalized text.
    """
    def normalize(example):
        example['sentence'] = normalise_text(example['sentence'])
        return example

    normalized = DatasetDict()
    for split, ds in tqdm(dataset.items(), desc="Normalizing text"):
        normalized[split] = ds.map(normalize)
    return normalized