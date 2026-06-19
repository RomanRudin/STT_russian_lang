from typing import Dict, Any
from acoustic.utils.registry import PluginRegistry

from .wer import compute_wer
from .cer import compute_cer
from.f1 import compute_f1
from .detailed_stats import compute_detailed_stats
from .ser import compute_ser
from .space_error_rate import compute_space_wer


# Define a dummy interface for signature checking
def _metric_interface(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    ...


METRICS = PluginRegistry("metric", interface=_metric_interface)

# Manual registration
METRICS.register("wer", compute_wer)
METRICS.register("cer", compute_cer)
METRICS.register("f1", compute_f1)
METRICS.register("detailed_stats", compute_detailed_stats)
METRICS.register("ser", compute_ser)
METRICS.register("space_wer", compute_space_wer)


def get_metric(name: str):
    return METRICS.get(name)