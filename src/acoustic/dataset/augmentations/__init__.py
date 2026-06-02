from acoustic.utils.registry import PluginRegistry

AUGMENTATIONS = PluginRegistry("augmentation")

def get_augmentation(name: str):
    """Return an augmentation function by name."""
    return AUGMENTATIONS.get(name)

# Register augmentations
from .noise_speed import apply as noise_speed_augment
AUGMENTATIONS.register("noise_speed", noise_speed_augment)