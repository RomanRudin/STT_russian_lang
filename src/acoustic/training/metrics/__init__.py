from .wer import compute_wer
from .cer import compute_cer

METRICS = {
    "wer": compute_wer,
    "cer": compute_cer,
}

def get_metric(name: str):
    """Return metric function by name."""
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS.keys())}")
    return METRICS[name]