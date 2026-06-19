import logging
from typing import Any, Dict
from datasets import DatasetDict, concatenate_datasets

from .fleurs_dataset import create_fleurs
from .golos_dataset import create_golos

logger = logging.getLogger(__name__)

def create_combined(cfg: Dict[str, Any]) -> DatasetDict:
    """
    Build a combined dataset from enabled sources, but only train and validation.
    Test split is completely ignored.
    """
    sources = cfg['dataset']['sources']
    train_parts = []
    val_parts = []

    # ---- FLEURS -------------------------------------------------------------
    if sources.get('fleurs', {}).get('enabled', False):
        logger.info("Adding FLEURS dataset (train + validation only)")
        fleurs = create_fleurs(cfg)
        train_parts.append(fleurs['train'])
        val_parts.append(fleurs['validation'])
        # test split from fleurs is deliberately ignored

    # ---- Golos --------------------------------------------------------------
    if sources.get('golos', {}).get('enabled', False):
        logger.info("Adding Golos dataset (train + validation only)")
        golos = create_golos(cfg)
        train_parts.append(golos['train'])
        val_parts.append(golos['validation'])
        # even if golos contains 'test', it's not used

    if not train_parts:
        raise ValueError("No data sources enabled. Enable at least one source in config.")

    # Return only train and validation
    merged = DatasetDict({
        "train": concatenate_datasets(train_parts),
        "validation": concatenate_datasets(val_parts),
    })
    logger.info("Combined dataset sizes: train=%d, validation=%d",
                len(merged["train"]), len(merged["validation"]))

    return merged