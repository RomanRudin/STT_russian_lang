import evaluate
from typing import Dict, Any

_cer_metric = evaluate.load("cer")

def compute_cer(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute Character Error Rate.

    Args:
        eval_pred: Dictionary with 'predictions' and 'references' lists of strings.

    Returns:
        Dictionary with 'cer' key.
    """
    preds = eval_pred.get("predictions", [])
    refs = eval_pred.get("references", [])
    if not refs:
        return {"cer": 1.0}
    cer = _cer_metric.compute(predictions=preds, references=refs)
    return {"cer": round(cer, 4)}