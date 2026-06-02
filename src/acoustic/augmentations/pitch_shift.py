import torch
import torchaudio
import random
from .base import WaveformAugmentation


class PitchShift(WaveformAugmentation):
    def __init__(self, min_semitones: float = -4, max_semitones: float = 4,
                 prob: float = 0.3):
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.prob = prob

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.prob:
            return waveform
        semitones = random.uniform(self.min_semitones, self.max_semitones)
        original_shape = waveform.shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        waveform = torchaudio.functional.pitch_shift(waveform, sample_rate, semitones)
        if len(original_shape) == 1:
            waveform = waveform.squeeze(0)
        return waveform