import os
import logging
from typing import Dict, Any
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

from acoustic.dataset.filters import get_filter
from acoustic.dataset.create_dataset import get_dataset_builder

logger = logging.getLogger(__name__)


def load_and_prepare_dataset(cfg: Dict[str, Any]) -> DatasetDict:
    cache_cfg = cfg['dataset'].get('cache', {})
    use_cache = cache_cfg.get('use_cache', True)
    cache_dir = cache_cfg.get('cache_dir', './data/processed')
    force_recreate = cache_cfg.get('force_recreate', False)

    if use_cache and not force_recreate and os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        logger.info("Loading raw dataset from cache: %s", cache_dir)
        dataset = load_from_disk(cache_dir)
        if 'audio' in dataset['train'].column_names and 'sentence' in dataset['train'].column_names:
            return dataset
        else:
            logger.warning("Cached dataset missing audio/sentence, recreating...")

    create_fn_str = cfg['dataset']['create_function']
    builder = get_dataset_builder(create_fn_str)
    dataset = builder(cfg)

    # Apply filters
    filter_names = cfg['dataset'].get('filters', [])
    for name in tqdm(filter_names, desc="Applying filters"):
        filter_fn = get_filter(name)
        dataset = filter_fn(dataset, cfg)

    # Optional sample limits
    max_train = cfg['dataset'].get('max_train_samples')
    max_eval = cfg['dataset'].get('max_eval_samples')

    if max_train is not None and max_train > 0:
        train_size = len(dataset['train'])
        if train_size > max_train:
            logger.info(f"Limiting train dataset to {max_train} samples (was {train_size})")
            dataset['train'] = dataset['train'].select(range(max_train))
        else:
            logger.info(f"Train dataset has {train_size} samples, which is <= max_train_samples={max_train}; keeping all.")

    if max_eval is not None and max_eval > 0:
        eval_size = len(dataset['validation'])
        if eval_size > max_eval:
            logger.info(f"Limiting validation dataset to {max_eval} samples (was {eval_size})")
            dataset['validation'] = dataset['validation'].select(range(max_eval))
        else:
            logger.info(f"Validation dataset has {eval_size} samples, which is <= max_eval_samples={max_eval}; keeping all.")

    # Save fully processed dataset to cache
    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(cache_dir)
    logger.info("Processed dataset saved to cache: %s", cache_dir)

    return dataset