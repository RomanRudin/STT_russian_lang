from typing import List, Callable, Dict, Any
from acoustic.training.metrics import get_metric

def load_metrics(metric_names: List[str]) -> List[Callable[[Dict[str, Any]], Dict[str, float]]]:
    """
    Load metric functions by their names.

    Args:
        metric_names: List of metric names (e.g., ["wer", "cer"]).

    Returns:
        List of metric callables.
    """
    return [get_metric(name) for name in metric_names]