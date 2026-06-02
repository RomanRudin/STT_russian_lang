"""
Whisper-specific data collator: pads audio, creates label tensors.
Accepts configurable max lengths and optional waveform augmentations.
"""

from typing import Dict, Any, List, Optional
import torch
from transformers import WhisperProcessor


class WhisperDataCollator:
    """
    Pads audio to `max_audio_length` seconds and builds label tensors.
    """

    def __init__(
        self,
        processor: WhisperProcessor,
        max_audio_length: int = 30,
        max_label_length: int = 225,
        augmentations: Optional[List] = None,
        **kwargs
    ):
        self.processor = processor
        self.max_audio_samples = max_audio_length * 16000
        self.max_label_length = max_label_length
        self.augmentations = augmentations or []

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [f["audio"]["array"] for f in features]
        sentences = [f["sentence"] for f in features]

        # Apply waveform augmentations if any
        if self.augmentations:
            augmented_audio = []
            for arr in audio_arrays:
                waveform = torch.tensor(arr).float()
                # Ensure waveform is (1, samples) for augmentations expecting channels
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                for aug in self.augmentations:
                    waveform = aug(waveform, sample_rate=16000)
                # Convert back to numpy (samples,)
                augmented_audio.append(waveform.squeeze(0).numpy())
            audio_arrays = augmented_audio

        # Feature extraction
        inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_audio_samples,
            truncation=True,
        )

        # Tokenize labels
        with self.processor.tokenizer.as_target_tokenizer():
            labels = self.processor.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_label_length,
            ).input_ids

        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        return {"input_features": inputs.input_features, "labels": labels}