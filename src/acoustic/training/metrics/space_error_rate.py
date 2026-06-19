from typing import Dict, Any
from jiwer import process_words

def compute_space_wer(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    preds = eval_pred.get("predictions", [])
    refs = eval_pred.get("references", [])
    if not refs:
        return {"space_wer": 1.0}

    def to_space_string(s: str) -> str:
        return ''.join(' ' if c == ' ' else 'a' for c in s)

    preds_space = [to_space_string(p) for p in preds]
    refs_space = [to_space_string(r) for r in refs]

    results = process_words(refs_space, preds_space)
    return {"space_wer": round(results.wer, 4)}