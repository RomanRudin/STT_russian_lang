from acoustic.utils.registry import PluginRegistry

COLLATORS = PluginRegistry("collator")

def get_collator_class(name: str):
    """Return a collator class by its registry name."""
    return COLLATORS.get(name)

# Manual registration
from .whisper_collator import WhisperDataCollator

COLLATORS.register("whisper", WhisperDataCollator)