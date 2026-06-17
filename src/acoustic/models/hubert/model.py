import torch
from transformers import HubertForCTC, Wav2Vec2Processor

def build_hubert(cfg):
    processor_name = cfg['model'].get('name', 'jonatasgrosman/wav2vec2-large-xlsr-53-russian')
    processor = Wav2Vec2Processor.from_pretrained(processor_name)
    
    model = HubertForCTC.from_pretrained(
        "facebook/hubert-large-ls960-ft",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )
    
    model.freeze_feature_encoder()
    
    return model, processor

def _hubert_generate(model, input_features, processor, **kwargs):
    outputs = model(input_features)
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return predicted_ids