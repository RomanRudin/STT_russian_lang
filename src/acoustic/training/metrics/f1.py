from typing import Dict, Any
from jiwer import process_words

def compute_f1(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    preds = eval_pred.get("predictions", [])
    refs = eval_pred.get("references", [])
    if not refs:
        return {"f1": 0.0}

    results = process_words(refs, preds)
    hits = results.hits
    substitutions = results.substitutions
    deletions = results.deletions
    insertions = results.insertions

    precision = hits / (hits + substitutions + insertions) if (hits + substitutions + insertions) else 0.0
    recall = hits / (hits + substitutions + deletions) if (hits + substitutions + deletions) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"f1": round(f1, 4)}