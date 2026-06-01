"""
Whisper model builder with LoRA support and data collator.
"""

import logging
from typing import Tuple, Dict, Any, Optional, List
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)


class WhisperDataCollator:
    """
    Data collator for Whisper models.
    Pads audio to 30 seconds (max length) to ensure mel features have fixed time dimension.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [f["audio"]["array"] for f in features]
        sentences = [f["sentence"] for f in features]

        # Pad all audio to 30 seconds (480000 samples at 16kHz)
        inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",      # pad to max_length below
            max_length=30 * 16000,     # 30 seconds at 16kHz
            truncation=True,
        )

        with self.processor.tokenizer.as_target_tokenizer():
            labels = self.processor.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=225,
            ).input_ids

        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        return {
            "input_features": inputs.input_features,  # shape (batch, 80, 3000)
            "labels": labels,
        }


def _apply_lora(model: WhisperForConditionalGeneration, r: int = 32, alpha: int = 64) -> WhisperForConditionalGeneration:
    """Apply LoRA adapters to the model."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError("Install peft: pip install peft")
    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_whisper(cfg: Dict[str, Any]) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Build Whisper model and processor from configuration.

    Args:
        cfg: Configuration dictionary with model.name, language, task, use_lora.

    Returns:
        Tuple of (model, processor).
    """
    model_name = cfg['model']['name']
    language = cfg['model']['language']
    task = cfg['model']['task']
    use_lora = cfg['model'].get('use_lora', False)

    logger.info("Loading processor from %s ...", model_name)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)

    logger.info("Loading model from %s ...", model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Set forced decoder IDs for language/task
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.config.suppress_tokens = []
    model.config.use_cache = False

    if use_lora:
        logger.info("Applying LoRA adapters...")
        model = _apply_lora(model)
    else:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: total=%d, trainable=%d (%.1f%%)",
                total_params, trainable_params, 100.0 * trainable_params / total_params)

    return model, processor


def get_data_collator(processor: WhisperProcessor) -> WhisperDataCollator:
    """
    Return the data collator for Whisper.

    Args:
        processor: WhisperProcessor instance.

    Returns:
        WhisperDataCollator instance.
    """
    return WhisperDataCollator(processor)


def load_checkpoint(checkpoint_dir: str, cfg: Dict[str, Any]) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load model and processor from checkpoint."""
    logger.info("Loading checkpoint from %s", checkpoint_dir)
    processor = WhisperProcessor.from_pretrained(checkpoint_dir)

    use_lora = cfg['model'].get('use_lora', False)
    if use_lora:
        from peft import PeftModel
        base_model_name = cfg['model']['name']
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)

    return model, processor