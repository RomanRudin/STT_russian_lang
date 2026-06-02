import logging
from typing import Tuple, Dict, Any
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)


def _apply_lora(model: WhisperForConditionalGeneration, r: int = 32, alpha: int = 64) -> WhisperForConditionalGeneration:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError("Install peft: pip install peft")
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
    model_name = cfg['model']['name']
    language = cfg['model']['language']
    task = cfg['model']['task']
    use_lora = cfg['model'].get('use_lora', False)

    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.config.suppress_tokens = []
    model.config.use_cache = False

    if use_lora:
        model = _apply_lora(model)
    else:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: total=%d, trainable=%d (%.1f%%)",
                total_params, trainable_params, 100.0 * trainable_params / total_params)

    return model, processor


def load_checkpoint(checkpoint_dir: str, cfg: Dict[str, Any]) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    logger.info("Loading checkpoint from %s", checkpoint_dir)
    processor = WhisperProcessor.from_pretrained(checkpoint_dir)
    use_lora = cfg['model'].get('use_lora', False)
    if use_lora:
        from peft import PeftModel
        base_model = WhisperForConditionalGeneration.from_pretrained(cfg['model']['name'])
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
    return model, processor