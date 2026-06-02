from typing import Dict, Any
from acoustic.utils.registry import PluginRegistry

from .wer import compute_wer
from .cer import compute_cer


# Define a dummy interface for signature checking
def _metric_interface(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    ...


METRICS = PluginRegistry("metric", interface=_metric_interface)

# Manual registration
METRICS.register("wer", compute_wer)
METRICS.register("cer", compute_cer)


def get_metric(name: str):
    return METRICS.get(name)