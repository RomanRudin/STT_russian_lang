import importlib
import os
import logging
from typing import Dict, Any
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

from acoustic.dataset.filters import get_filter

logger = logging.getLogger(__name__)


def load_and_prepare_dataset(cfg: Dict[str, Any]) -> DatasetDict:
    """
    Load or create a raw dataset (audio + sentence) with caching and sample limits.
    """
    cache_cfg = cfg['dataset'].get('cache', {})
    use_cache = cache_cfg.get('use_cache', True)
    cache_dir = cache_cfg.get('cache_dir', './data/processed')
    force_recreate = cache_cfg.get('force_recreate', False)

    # Try to load from cache
    if use_cache and not force_recreate and os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        logger.info("Loading raw dataset from cache: %s", cache_dir)
        dataset = load_from_disk(cache_dir)
        if 'audio' in dataset['train'].column_names and 'sentence' in dataset['train'].column_names:
            # Если в кэше уже есть данные, но мы хотим применить ограничения,
            # то их нужно применить и к загруженному датасету (без пересохранения).
            # Для простоты предлагаем не делать этого — пользователь должен
            # либо изменить force_recreate, либо удалить кэш.
            return dataset
        else:
            logger.warning("Cached dataset missing audio/sentence, recreating...")

    # Build from scratch using the configured create_function
    create_fn_str = cfg['dataset']['create_function']
    if create_fn_str == "combined_dataset.create_combined":
        from acoustic.dataset.create_dataset.combined_dataset import create_combined
        dataset = create_combined(cfg)
    else:
        module_name, func_name = create_fn_str.rsplit('.', 1)
        full_module_path = f"acoustic.dataset.create_dataset.{module_name}"
        module = importlib.import_module(full_module_path)
        create_fn = getattr(module, func_name)
        dataset = create_fn(cfg)

    # Apply filters (duration, text normalization)
    filter_names = cfg['dataset'].get('filters', [])
    for name in tqdm(filter_names, desc="Applying filters"):
        filter_fn = get_filter(name)
        dataset = filter_fn(dataset, cfg)

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

    # Save raw dataset to cache
    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(cache_dir)
    logger.info("Raw dataset saved to cache: %s", cache_dir)

    return dataset