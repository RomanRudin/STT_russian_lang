import evaluate
from typing import Dict, Any

_wer_metric = evaluate.load("wer")

def compute_wer(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute Word Error Rate.

    Args:
        eval_pred: Dictionary with 'predictions' and 'references' lists of strings.

    Returns:
        Dictionary with 'wer' key.
    """
    preds = eval_pred.get("predictions", [])
    refs = eval_pred.get("references", [])
    if not refs:
        return {"wer": 1.0}
    wer = _wer_metric.compute(predictions=preds, references=refs)
    return {"wer": round(wer, 4)}