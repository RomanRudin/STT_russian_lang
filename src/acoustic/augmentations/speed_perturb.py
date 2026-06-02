import torch
import torchaudio
import random
from .base import WaveformAugmentation


class SpeedPerturb(WaveformAugmentation):
    def __init__(self, speed_range: list = None, prob: float = 0.2):
        if speed_range is None:
            speed_range = [0.9, 1.1]
        self.speed_range = speed_range
        self.prob = prob

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.prob:
            return waveform
        factor = random.uniform(*self.speed_range)
        # Ensure waveform has at least one channel dimension: (channels, samples)
        original_shape = waveform.shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        elif waveform.dim() > 2:
            # If batch dim exists, assume we want to process per item
            raise ValueError("SpeedPerturb expects waveform of shape (channels, samples) or (samples,)")

        orig_freq = sample_rate
        new_freq = int(orig_freq * factor)
        waveform = torchaudio.functional.resample(waveform, orig_freq, new_freq)
        waveform = torchaudio.functional.resample(waveform, new_freq, orig_freq)

        # Restore original dimensionality
        if len(original_shape) == 1:
            waveform = waveform.squeeze(0)
        return waveform