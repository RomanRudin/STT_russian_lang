from datasets import DatasetDict
from acoustic.utils.registry import PluginRegistry

from .fleurs_dataset import create_fleurs
from .golos_dataset import create_golos
from .combined_dataset import create_combined


def _builder_interface(cfg: dict) -> DatasetDict:
    ...


DATASET_BUILDERS = PluginRegistry("dataset_builder", interface=_builder_interface)

# Manual registration
DATASET_BUILDERS.register("fleurs", create_fleurs)
DATASET_BUILDERS.register("golos", create_golos)
DATASET_BUILDERS.register("combined_dataset", create_combined)


def get_dataset_builder(name: str):
    """
    Return a dataset builder function by its registered name.
    For backward compatibility, if `name` contains a dot (e.g., "combined_dataset.create_combined"),
    the part after the dot is ignored and only the first part is used as the registry key.
    """
    # Support old-style "combined_dataset.create_combined" by using the part before the dot
    if '.' in name:
        name = name.split('.')[0]
    return DATASET_BUILDERS.get(name)