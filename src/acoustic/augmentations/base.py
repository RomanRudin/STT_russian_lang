from abc import ABC, abstractmethod
import torch


class WaveformAugmentation(ABC):
    @abstractmethod
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Args:
            waveform: (batch, channels, samples) or (channels, samples) or (samples,)?
            sample_rate: int
        Returns:
            augmented waveform with same shape
        """
        pass