"""
metrics.py
==========
Метрики для оценки качества восстановления пунктуации, регистра и абзацев.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.labels import PUNCT_LABELS, CASE_LABELS, PARA_LABELS, IGNORE_INDEX


def compute_token_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    labels: List[str] = None,
) -> Dict[str, float]:
    """Вычисляет accuracy и macro F1 для одного набора меток."""
    mask = targets != ignore_index
    y_true = targets[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()

    if len(y_true) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0}

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {"accuracy": acc, "f1_macro": f1}

    # Per-class метрики (опционально)
    if labels is not None:
        report = classification_report(
            y_true, y_pred, labels=range(len(labels)), target_names=labels,
            zero_division=0, output_dict=True
        )
        for cls_name, cls_metrics in report.items():
            # Пропускаем служебные ключи и проверяем, что cls_metrics - словарь
            if cls_name in ["accuracy", "macro avg", "weighted avg"]:
                continue
            if isinstance(cls_metrics, dict) and "f1_score" in cls_metrics:
                metrics[f"f1_{cls_name}"] = cls_metrics["f1_score"]

    return metrics


def compute_all_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Вычисляет все метрики для батча:
      - punct_accuracy, punct_f1_macro, и per-class F1 для пунктуации
      - case_accuracy, case_f1_macro
      - para_accuracy, para_f1_macro
    """
    metrics = {}
    
    # Пунктуация
    punct_preds = outputs["punct"].argmax(dim=-1)
    punct_targets = batch["punct"]
    punct_metrics = compute_token_metrics(
        punct_preds, punct_targets, ignore_index=IGNORE_INDEX,
        labels=PUNCT_LABELS
    )
    for k, v in punct_metrics.items():
        metrics[f"punct_{k}"] = v
    
    # Регистр
    case_preds = outputs["case"].argmax(dim=-1)
    case_targets = batch["case"]
    case_metrics = compute_token_metrics(
        case_preds, case_targets, ignore_index=IGNORE_INDEX,
        labels=CASE_LABELS
    )
    for k, v in case_metrics.items():
        metrics[f"case_{k}"] = v
    
    # Абзацы
    para_preds = outputs["para"].argmax(dim=-1)
    para_targets = batch["para"]
    para_metrics = compute_token_metrics(
        para_preds, para_targets, ignore_index=IGNORE_INDEX,
        labels=PARA_LABELS
    )
    for k, v in para_metrics.items():
        metrics[f"para_{k}"] = v
    
    return metrics
