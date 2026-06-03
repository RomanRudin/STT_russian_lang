import torch
import random
from .base import WaveformAugmentation

class Gain(WaveformAugmentation):
    """Случайное усиление/ослабление громкости."""
    def __init__(self, min_db: float = -10.0, max_db: float = 5.0, prob: float = 0.5):
        self.min_db = min_db
        self.max_db = max_db
        self.prob = prob

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.prob:
            return waveform
        db = random.uniform(self.min_db, self.max_db)
        gain_linear = 10 ** (db / 20.0)
        return waveform * gain_linear