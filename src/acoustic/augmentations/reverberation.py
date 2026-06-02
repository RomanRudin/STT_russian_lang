import random
import torch
from .base import WaveformAugmentation

try:
    from torch_audiomentations import ApplyImpulseResponse
    TORCH_AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    TORCH_AUDIOMENTATIONS_AVAILABLE = False


class ReverbAugmentation(WaveformAugmentation):
    def __init__(self, rir_path: str = None, prob: float = 0.2):
        self.prob = prob
        if not TORCH_AUDIOMENTATIONS_AVAILABLE:
            raise ImportError(
                "torch-audiomentations is required for reverb augmentation. "
                "Install it with: pip install torch-audiomentations"
            )
        # Use default RIRs if no path provided (built-in small set)
        self.apply_ir = ApplyImpulseResponse(
            ir_paths=[rir_path] if rir_path else None,  # None uses built-in
            p=1.0,  # we'll control probability ourselves
        )

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.prob:
            return waveform
        # ApplyImpulseResponse expects (batch, channels, samples)
        original_shape = waveform.shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # (1, channels, samples)
        else:
            # already has batch dim? assume it's correct
            pass
        augmented = self.apply_ir(waveform, sample_rate)
        # Restore original shape
        if len(original_shape) == 1:
            augmented = augmented.squeeze(0).squeeze(0)  # (samples,)
        elif len(original_shape) == 2:
            augmented = augmented.squeeze(0)  # (channels, samples)
        return augmented