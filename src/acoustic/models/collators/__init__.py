from acoustic.utils.registry import PluginRegistry

COLLATORS = PluginRegistry("collator")

def get_collator_class(name: str):
    """Return a collator class by its registry name."""
    return COLLATORS.get(name)

# Manual registration
from .whisper_collator import WhisperDataCollator
from .wav2vec2_collator import Wav2Vec2DataCollator
from .custom_CNN_RNN_collator import  CustomSTTCollator

from .correction_collator import  RuT5DataCollator
from .lstm_correction_collator import LSTMCorrectionCollator


COLLATORS.register("whisper", WhisperDataCollator)
COLLATORS.register("wav2vec2", Wav2Vec2DataCollator)
COLLATORS.register("custom_CNN_RNN", CustomSTTCollator)

COLLATORS.register("correction_model", RuT5DataCollator)
COLLATORS.register("lstm_correction", LSTMCorrectionCollator)
