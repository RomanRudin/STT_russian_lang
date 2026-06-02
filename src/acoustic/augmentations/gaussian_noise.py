import torch
import random
from .base import WaveformAugmentation


class GaussianNoise(WaveformAugmentation):
    def __init__(self, noise_level_min: float = 0.001, noise_level_max: float = 0.015,
                 prob: float = 0.3):
        self.noise_level_min = noise_level_min
        self.noise_level_max = noise_level_max
        self.prob = prob

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.prob:
            return waveform
        noise_level = random.uniform(self.noise_level_min, self.noise_level_max)
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise