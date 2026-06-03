from typing import Dict, Any

def compute_ser(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    """
    Sentence Error Rate: fraction of sentences with at least one error.
    """
    preds = eval_pred.get("predictions", [])
    refs = eval_pred.get("references", [])
    if not refs:
        return {"ser": 1.0}

    errors = sum(1 for p, r in zip(preds, refs) if p != r)
    ser = errors / len(refs)
    return {"ser": round(ser, 4)}