from typing import Dict, Any
from jiwer import compute_measures

def compute_space_wer(eval_pred: Dict[str, Any]) -> Dict[str, float]:
    """
    Space Error Rate: WER on strings where only spaces are preserved as ' ' and
    all other characters are mapped to 'a'.
    """
    preds = eval_pred.get("predictions", [])
    refs = eval_pred.get("references", [])
    if not refs:
        return {"space_wer": 1.0}

    def to_space_string(s: str) -> str:
        return ''.join(' ' if c == ' ' else 'a' for c in s)

    preds_space = [to_space_string(p) for p in preds]
    refs_space = [to_space_string(r) for r in refs]

    measures = compute_measures(refs_space, preds_space)
    return {"space_wer": round(measures["wer"], 4)}