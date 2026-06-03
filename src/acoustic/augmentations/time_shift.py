import torch
import random
from .base import WaveformAugmentation

class TimeShift(WaveformAugmentation):
    """Сдвиг сигнала во времени с заполнением нулями."""
    def __init__(self, max_shift_ratio: float = 0.1, prob: float = 0.5):
        self.max_shift_ratio = max_shift_ratio
        self.prob = prob

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.prob:
            return waveform

        original_shape = waveform.shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        channels, samples = waveform.shape

        shift = int(random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * samples)
        if shift == 0:
            return waveform.squeeze(0) if len(original_shape) == 1 else waveform

        if shift > 0:
            waveform = torch.cat([
                torch.zeros(channels, shift, device=waveform.device),
                waveform[:, :-shift]
            ], dim=-1)
        else:
            shift_abs = -shift
            waveform = torch.cat([
                waveform[:, shift_abs:],
                torch.zeros(channels, shift_abs, device=waveform.device)
            ], dim=-1)

        if len(original_shape) == 1:
            waveform = waveform.squeeze(0)
        return waveform