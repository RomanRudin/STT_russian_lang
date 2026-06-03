from typing import Dict, Any
from jiwer import compute_measures

def compute_f1(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute micro-averaged word-level F1 score.
    """
    preds = eval_pred.get("predictions", [])
    refs = eval_pred.get("references", [])
    if not refs:
        return {"f1": 0.0}

    measures = compute_measures(refs, preds)
    hits = measures["hits"]
    substitutions = measures["substitutions"]
    deletions = measures["deletions"]
    insertions = measures["insertions"]

    precision = hits / (hits + substitutions + insertions) if (hits + substitutions + insertions) else 0.0
    recall = hits / (hits + substitutions + deletions) if (hits + substitutions + deletions) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"f1": round(f1, 4)}