from acoustic.utils.registry import PluginRegistry
from .base import WaveformAugmentation

AUGMENTATION_CLASSES = PluginRegistry("augmentation")


def build_augmentations(config: dict):
    """
    Build a list of augmentation instances from a config dict.
    config: dict of {aug_name: params} where params must contain 'enabled': true/false.
    Returns list of WaveformAugmentation instances.
    """
    instances = []
    if not config:
        return instances
    for name, params in config.items():
        if isinstance(params, bool):
            params = {"enabled": params}
        if not params.get("enabled", False):
            continue
        aug_cls = AUGMENTATION_CLASSES.get(name)
        # Remove 'enabled' from params before passing to constructor
        aug_params = {k: v for k, v in params.items() if k != "enabled"}
        instances.append(aug_cls(**aug_params))
    return instances


# Import and register implementations
from .gaussian_noise import GaussianNoise
from .speed_perturb import SpeedPerturb
from .pitch_shift import PitchShift
from .reverberation import ReverbAugmentation
from .gain import Gain
from .time_shift import TimeShift

AUGMENTATION_CLASSES.register("gaussian_noise", GaussianNoise)
AUGMENTATION_CLASSES.register("speed_perturb", SpeedPerturb)
AUGMENTATION_CLASSES.register("pitch_shift", PitchShift)
AUGMENTATION_CLASSES.register("reverb", ReverbAugmentation)
AUGMENTATION_CLASSES.register("gain", Gain)
AUGMENTATION_CLASSES.register("time_shift", TimeShift)