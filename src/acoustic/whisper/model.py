"""
src/acoustic/model.py

Whisper model wrapper for Russian STT fine-tuning.

Supports two fine-tuning modes:
  1. Full fine-tuning  — all weights updated (needs ~40 GB VRAM for large-v3)
  2. LoRA fine-tuning  — only low-rank adapters updated (~8 GB VRAM)

Usage:
    model, processor = build_model(cfg)
"""

import logging
from typing import Optional, Tuple

import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logger = logging.getLogger(__name__)


def _apply_lora(model: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Install peft to use LoRA fine-tuning:  pip install peft"
        ) from e

    # Freeze everything 
    for param in model.parameters():
        param.requires_grad = False

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        # Target the query & value projections in every attention layer
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        # Whisper is seq2seq, but peft treats the whole model as FEATURE_EXTRACTION, so when task_type is not SEQ2SEQ — use FEATURE_EXTRACTION to avoid head issues
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def build_model(
    cfg: dict,
    use_lora: bool = False,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    model_name: str = cfg["model"]["name"]
    language:   str = cfg["model"]["language"]    # "russian"
    task:       str = cfg["model"]["task"]         # "transcribe"

    logger.info("Loading processor from %s …", model_name)
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task=task,
    )

    logger.info("Loading model from %s …", model_name)
    model: WhisperForConditionalGeneration = (
        WhisperForConditionalGeneration.from_pretrained(model_name)
    )

    # Hard-wire language & task tokens so the model never tries to auto-detect
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task=task
    )
    # Suppress these tokens during beam search generation
    model.config.suppress_tokens = []

    # Disable the gradient checkpointing flag that Whisper sets by default
    # for generation (we'll enable it separately for training if needed)
    model.config.use_cache = False

    if use_lora:
        logger.info("Applying LoRA adapters …")
        model = _apply_lora(model)
    else:
        # Full fine-tune — gradient checkpointing saves VRAM at cost of speed
        model.gradient_checkpointing_enable()

    _log_param_counts(model)
    return model, processor

def _log_param_counts(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Parameters — total: %s  trainable: %s  (%.1f %%)",
        f"{total:,}",
        f"{trainable:,}",
        100.0 * trainable / total,
    )


def load_checkpoint(
    checkpoint_dir: str,
    cfg: dict,
    use_lora: bool = False,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    logger.info("Loading checkpoint from %s …", checkpoint_dir)
    processor = WhisperProcessor.from_pretrained(checkpoint_dir)

    if use_lora:
        from peft import PeftModel  # type: ignore

        base_name = cfg["model"]["name"]
        base = WhisperForConditionalGeneration.from_pretrained(base_name)
        model = PeftModel.from_pretrained(base, checkpoint_dir)
        # Merge LoRA weights into base for faster inference
        model = model.merge_and_unload()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)

    return model, processor
