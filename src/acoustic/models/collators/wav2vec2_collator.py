import torch
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import Wav2Vec2Processor

class Wav2Vec2DataCollator:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        augmentations: Optional[List] = None,
        **kwargs
    ):
        self.processor = processor
        self.augmentations = augmentations or []

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [f["audio"]["array"] for f in features]
        sentences = [f["sentence"] for f in features]

        if self.augmentations:
            augmented_audio = []
            for arr in audio_arrays:
                waveform = torch.tensor(arr).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                for aug in self.augmentations:
                    waveform = aug(waveform, sample_rate=16000)
                augmented_audio.append(waveform.squeeze(0).numpy())
            audio_arrays = augmented_audio

        batch = self.processor(
            audio_arrays,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer(
            sentences,
            padding=True,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch