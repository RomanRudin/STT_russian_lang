import random
import torch
import torchaudio

class AudioAugmenter:
    """Light on-the-fly augmentation for 16 kHz waveform tensors (1, T)."""

    def __init__(self, cfg: dict):
        self.noise_prob = cfg.get("noise_prob", 0.3)
        self.noise_min = cfg.get("noise_level_min", 0.001)
        self.noise_max = cfg.get("noise_level_max", 0.015)
        self.speed_prob = cfg.get("speed_prob", 0.2)
        self.speed_range = cfg.get("speed_range", [0.9, 1.1])

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() < self.noise_prob:
            wav = wav + torch.randn_like(wav) * random.uniform(self.noise_min, self.noise_max)
        if random.random() < self.speed_prob:
            factor = random.uniform(*self.speed_range)
            wav = torchaudio.functional.resample(wav, int(16000 * factor), 16000)
        return torch.clamp(wav, -1.0, 1.0)