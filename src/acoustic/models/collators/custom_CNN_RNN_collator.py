import torch
import torchaudio
from typing import Dict, Any, List, Optional

class CustomSTTCollator:
    """Data collator for STT: applies augmentations, generates Mel-spectrograms, and pads inputs."""
    def __init__(
        self,
        tokenizer,
        sample_rate: int = 16000,
        n_mels: int = 80,
        augmentations: Optional[List] = None,
    ):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.augmentations = augmentations or []
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [f["audio"]["array"] for f in features]
        sentences = [f["sentence"] for f in features]

        mel_spectrograms = []
        input_lengths = []
        
        for arr in audio_arrays:
            waveform = torch.tensor(arr).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            for aug in self.augmentations:
                waveform = aug(waveform, sample_rate=self.sample_rate)

            mel = self.mel_transform(waveform).squeeze(0)
            mel = torch.log(mel + 1e-9) 
            mel = (mel - mel.mean(dim=-1, keepdim=True)) / (mel.std(dim=-1, keepdim=True) + 1e-9)
            
            mel_spectrograms.append(mel)
            input_lengths.append(mel.shape[1])

        max_mel_len = max(input_lengths)
        padded_mels = torch.zeros(len(mel_spectrograms), mel_spectrograms[0].size(0), max_mel_len)
        for i, mel in enumerate(mel_spectrograms):
            padded_mels[i, :, :mel.shape[1]] = mel

        labels_batch = self.tokenizer(
            sentences,
            padding=True,
            return_tensors="pt"
        )
        
        labels = labels_batch["input_ids"]
        target_lengths = torch.sum(labels_batch["attention_mask"], dim=1)
        labels = labels.masked_fill(labels_batch["attention_mask"].ne(1), -100)

        return {
            "input_features": padded_mels,
            "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
            "labels": labels,
            "target_lengths": target_lengths
        }