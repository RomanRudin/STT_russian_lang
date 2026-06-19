import logging
from typing import Any, Dict
from datasets import DatasetDict, Audio, load_dataset

logger = logging.getLogger(__name__)

def create_fleurs(cfg: Dict[str, Any]) -> DatasetDict:
    sample_rate = cfg['dataset']['params']['sample_rate']
    logger.info("Loading FLEURS (ru_ru) from HuggingFace...")
    ds = load_dataset("google/fleurs", "ru_ru", trust_remote_code=True)
    # Берём только train и validation, test игнорируем
    ds = DatasetDict({
        "train": ds["train"],
        "validation": ds["validation"],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    if "transcription" in ds["train"].column_names:
        ds = ds.rename_column("transcription", "sentence")
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("audio", "sentence")])
    return ds