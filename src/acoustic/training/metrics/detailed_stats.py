from typing import Dict, Any
from jiwer import compute_measures

def compute_detailed_stats(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    """
    Return detailed edit statistics: hits, substitutions, deletions, insertions, WER.
    """
    preds = eval_pred.get("predictions", [])
    refs = eval_pred.get("references", [])
    if not refs:
        return {"hits": 0, "substitutions": 0, "deletions": 0, "insertions": 0, "wer": 1.0}

    measures = compute_measures(refs, preds)
    return {
        "hits": measures["hits"],
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "wer": round(measures["wer"], 4)   # дублирует основную метрику, можно убрать при желании
    }