from datasets import DatasetDict
from typing import Any, Dict
from tqdm import tqdm

def apply(dataset: DatasetDict, cfg: Dict[str, Any]) -> DatasetDict:
    """
    Filter examples by audio duration.

    Args:
        dataset: DatasetDict with 'audio' column.
        cfg: Configuration containing dataset.params.min_duration and max_duration.

    Returns:
        Filtered DatasetDict.
    """
    min_dur = cfg['dataset']['params'].get('min_duration', 0.0)
    max_dur = cfg['dataset']['params'].get('max_duration', float('inf'))

    def is_duration_ok(example):
        audio = example['audio']
        duration = len(audio['array']) / audio['sampling_rate']
        return min_dur <= duration <= max_dur

    filtered = DatasetDict()
    for split, ds in tqdm(dataset.items(), desc="Applying duration filter"):
        filtered[split] = ds.filter(is_duration_ok)
        print(f"{split}: kept {len(filtered[split])} / {len(ds)} examples")
    return filtered