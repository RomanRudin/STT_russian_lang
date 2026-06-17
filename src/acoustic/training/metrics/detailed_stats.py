from typing import Dict, Any
from jiwer import process_words

def compute_detailed_stats(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    preds = eval_pred.get("predictions", [])
    refs = eval_pred.get("references", [])
    if not refs:
        return {"hits": 0, "substitutions": 0, "deletions": 0, "insertions": 0, "wer": 1.0}

    results = process_words(refs, preds)
    return {
        "hits": results.hits,
        "substitutions": results.substitutions,
        "deletions": results.deletions,
        "insertions": results.insertions,
        "wer": round(results.wer, 4)
    }