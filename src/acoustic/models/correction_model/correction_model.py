import logging
from typing import Tuple, Dict, Any
from transformers import T5ForConditionalGeneration, AutoTokenizer

logger = logging.getLogger(__name__)


def build_rut5(cfg: Dict[str, Any]) -> Tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """Instantiate ruT5 model and tokenizer from config."""
    model_name = cfg['model'].get('name', 'cointegrated/rut5-small')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: total=%d, trainable=%d (%.1f%%)",
                total_params, trainable_params, 100.0 * trainable_params / total_params)

    return model, tokenizer

def _rut5_generate(model, input_data, processor):
    """Generate output token IDs from ruT5 model."""
    return model.generate(input_data)


def load_rut5_checkpoint(checkpoint_dir: str, cfg: Dict[str, Any]) -> Tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """Load a pretrained ruT5 model and tokenizer from a local checkpoint."""
    logger.info("Loading checkpoint from %s", checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
    return model, tokenizer