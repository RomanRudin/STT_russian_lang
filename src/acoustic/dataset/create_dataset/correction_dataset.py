import os
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from acoustic.utils.config import load_config
from acoustic.models.load_model import build_model
from acoustic.models import get_generate_method

logger = logging.getLogger(__name__)


def _load_acoustic_checkpoint(model, cfg):
    """Load trained weights from the best or final checkpoint of the acoustic model."""
    output_dir = cfg["training"]["output_dir"]
    final_dir = os.path.join(output_dir, "final_model")

    if os.path.isdir(final_dir):
        checkpoint_dir = final_dir
    else:
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1])
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints or final_model found in {output_dir}. Train the acoustic model first."
            )
        checkpoint_dir = os.path.join(output_dir, checkpoints[-1])

    logger.info("Loading acoustic weights from %s", checkpoint_dir)
    safetensor_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.isfile(safetensor_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensor_path)
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {checkpoint_dir}")

    model.load_state_dict(state_dict)
    logger.info("Acoustic checkpoint loaded successfully.")
    return model


def _run_acoustic_inference(acoustic_cfg_path, batch_size, num_workers, pin_memory, max_train):
    """Run acoustic model on the training set and return list of {stt_text, reference} dicts."""
    from acoustic.dataset.load_dataset import load_and_prepare_dataset

    cfg = load_config(acoustic_cfg_path)
    dataset = load_and_prepare_dataset(cfg)
    train_ds = dataset["train"]

    if max_train is not None and len(train_ds) > max_train:
        logger.info("Trimming acoustic dataset from %d to %d samples.", len(train_ds), max_train)
        train_ds = train_ds.select(range(max_train))

    logger.info("Loading acoustic model from %s", acoustic_cfg_path)
    model, processor, data_collator = build_model(cfg)
    model = _load_acoustic_checkpoint(model, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    generate_fn = get_generate_method(cfg["model"]["builder"])

    dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    reference_sentences = train_ds["sentence"]
    records = []
    ds_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Acoustic inference"):
            input_features = batch["input_features"].to(device)
            input_lengths = batch.get("input_lengths")
            if input_lengths is not None:
                input_lengths = input_lengths.to(device)

            predicted_ids = generate_fn(
                model, input_features, processor=processor, input_lengths=input_lengths
            )
            pred_texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            for pred_text in pred_texts:
                ref_text = reference_sentences[ds_idx]
                records.append(
                    {"stt_text": pred_text.strip(), "reference": ref_text.strip()}
                )
                ds_idx += 1

    return records


def create_correction_dataset(cfg: dict) -> DatasetDict:
    """Build a DatasetDict for the ruT5 text correction task."""
    cache_cfg = cfg["dataset"]["cache"]
    cache_dir = cache_cfg["cache_dir"]
    force_recreate = cache_cfg.get("force_recreate", False)

    # Try to load from cache
    if os.path.exists(cache_dir) and not force_recreate:
        logger.info("Loading correction dataset from cache: %s", cache_dir)
        return DatasetDict.load_from_disk(cache_dir)

    corr_cfg = cfg["dataset"]["sources"]["correction"]
    acoustic_cfg_path = corr_cfg["acoustic_config"]
    batch_size = corr_cfg.get("generation_batch_size", 32)
    num_workers = corr_cfg.get("generation_num_workers", 4)
    pin_memory = corr_cfg.get("generation_pin_memory", True)
    max_train = cfg["dataset"].get("max_train_samples")
    max_eval = cfg["dataset"].get("max_eval_samples")
    val_fraction = corr_cfg.get("val_fraction", 0.05)

    records = _run_acoustic_inference(
        acoustic_cfg_path, batch_size, num_workers, pin_memory, max_train
    )

    import random
    random.shuffle(records)

    # Filter out degenerate examples
    records = [r for r in records if len(r["reference"].strip()) > 1 and len(r["stt_text"].strip()) > 0]

    n_val = int(len(records) * val_fraction)
    val_records = records[:n_val]
    train_records = records[n_val:]

    if max_train and max_train < len(train_records):
        train_records = train_records[:max_train]
    if max_eval and max_eval < len(val_records):
        val_records = val_records[:max_eval]

    result = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
    })

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    result.save_to_disk(cache_dir)
    logger.info("Correction dataset created and cached. Splits: %s", {k: len(v) for k, v in result.items()})
    return result