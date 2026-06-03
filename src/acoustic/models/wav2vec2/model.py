import logging
from typing import Tuple, Dict, Any
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


logger = logging.getLogger(__name__)

def build_wav2vec2(cfg: Dict[str, Any]) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    model_name = cfg['model']['name']
    
    # Загружаем процессор (он включает в себя feature_extractor и tokenizer)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    # Загружаем модель для CTC
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    # Для файн-тюнинга Wav2Vec2 крайне рекомендуется замораживать сверточный feature extractor
    # Это экономит память и предотвращает переобучение на ранних этапах
    freeze_extractor = cfg['model'].get('freeze_feature_extractor', True)
    if freeze_extractor:
        model.freeze_feature_encoder()
        logger.info("Wav2Vec2 feature encoder is frozen.")

    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: total=%d, trainable=%d (%.1f%%)",
                total_params, trainable_params, 100.0 * trainable_params / total_params)

    return model, processor


def _wav2vec2_generate(model, input_data, processor):
    import torch
    outputs = model(input_data)
    logits = outputs.logits
    return torch.argmax(logits, dim=-1)