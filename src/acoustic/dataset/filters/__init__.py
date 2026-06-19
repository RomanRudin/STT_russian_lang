"""
acoustic/dataset/filters/__init__.py
Filter registry and convenience getter.
"""

from datasets import DatasetDict
from acoustic.utils.registry import PluginRegistry

from .duration_filter import apply as duration_filter
from .text_normalizer import apply as text_normalizer


# Define a dummy interface for signature checking
def _filter_interface(dataset: DatasetDict, cfg: dict) -> DatasetDict:
    ...


FILTERS = PluginRegistry("filter", interface=_filter_interface)

# Manual registration
FILTERS.register("duration_filter", duration_filter)
FILTERS.register("text_normalizer", text_normalizer)


def get_filter(name: str):
    return FILTERS.get(name)