"""
acoustic/training package.
Exports: get_trainer_class, load_metrics, get_callback.
"""

from acoustic.utils.registry import PluginRegistry
from .load_metrics import load_metrics
from .callbacks import get_callback

TRAINER_CLASSES = PluginRegistry("trainer_class")

def get_trainer_class(name: str):
    """Return a Trainer class by its registered name."""
    return TRAINER_CLASSES.get(name)

# Register available trainers
from .trainer import BaseTrainer
from .trainers.whisper_trainer import WhisperTrainer
from .trainers.wav2vec2_trainer import Wav2Vec2Trainer

from .trainers.custom_CNN_RNN_trainer import CustomSTTTrainer
from .trainers.correction_trainer import RuT5Trainer

TRAINER_CLASSES.register("BaseTrainer", BaseTrainer)
TRAINER_CLASSES.register("WhisperTrainer", WhisperTrainer)
TRAINER_CLASSES.register("Wav2Vec2Trainer", Wav2Vec2Trainer)

TRAINER_CLASSES.register("custom_CNN_RNN_Trainer", CustomSTTTrainer)
TRAINER_CLASSES.register("correction_Trainer", RuT5Trainer)
