"""
evaluate.py
===========
Оценка моделей: precision / recall / F1 по классам для каждой из трёх голов,
а также macro-F1 пунктуации (главная метрика, по которой выбираем лучшую
модель и сравниваем архитектуры).

Метрики считаются только по «значимым» позициям (метка != IGNORE_INDEX),
т.е. паддинг и субтокены-продолжения исключаются автоматически.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    IGNORE_INDEX, PUNCT_LABELS, PARA_LABELS, CAP_LABELS,
    NUM_PUNCT, NUM_PARA, NUM_CAP,
)


def _per_class_prf(preds: np.ndarray, golds: np.ndarray, num_classes: int):
    """Возвращает (precision[], recall[], f1[], support[]) по классам."""
    p = np.zeros(num_classes); r = np.zeros(num_classes)
    f = np.zeros(num_classes); sup = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        tp = int(np.sum((preds == c) & (golds == c)))
        fp = int(np.sum((preds == c) & (golds != c)))
        fn = int(np.sum((preds != c) & (golds == c)))
        sup[c] = int(np.sum(golds == c))
        p[c] = tp / (tp + fp) if (tp + fp) else 0.0
        r[c] = tp / (tp + fn) if (tp + fn) else 0.0
        f[c] = 2 * p[c] * r[c] / (p[c] + r[c]) if (p[c] + r[c]) else 0.0
    return p, r, f, sup


@torch.no_grad()
def evaluate(model, loader: DataLoader, device) -> Dict[str, float]:
    """
    Прогоняет loader, собирает предсказания трёх голов и считает метрики.
    Возвращает плоский словарь чисел (для логов/сравнения экспериментов).
    """
    model.eval()
    buckets = {h: {"pred": [], "gold": []} for h in ("punct", "para", "cap")}

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(**batch)
        for head in buckets:
            pred = logits[head].argmax(-1).reshape(-1).cpu().numpy()
            gold = batch[head].reshape(-1).cpu().numpy()
            keep = gold != IGNORE_INDEX
            buckets[head]["pred"].append(pred[keep])
            buckets[head]["gold"].append(gold[keep])

    specs = {
        "punct": (PUNCT_LABELS, NUM_PUNCT),
        "para": (PARA_LABELS, NUM_PARA),
        "cap": (CAP_LABELS, NUM_CAP),
    }
    metrics: Dict[str, float] = {}
    for head, (labels, num) in specs.items():
        preds = np.concatenate(buckets[head]["pred"]) if buckets[head]["pred"] else np.array([])
        golds = np.concatenate(buckets[head]["gold"]) if buckets[head]["gold"] else np.array([])
        if golds.size == 0:
            metrics[f"{head}_f1_macro"] = 0.0
            continue
        p, r, f, sup = _per_class_prf(preds, golds, num)
        # macro по классам, исключая «пустой» класс (O / NO_PARA / LOWER),
        # чтобы метрика отражала качество именно на знаках, а не на доминанте.
        sig = list(range(1, num))
        metrics[f"{head}_f1_macro"] = float(np.mean([f[c] for c in sig])) if sig else float(f.mean())
        metrics[f"{head}_acc"] = float(np.mean(preds == golds))
        for c, name in enumerate(labels):
            metrics[f"{head}/{name}_f1"] = float(f[c])
            metrics[f"{head}/{name}_p"] = float(p[c])
            metrics[f"{head}/{name}_r"] = float(r[c])
            metrics[f"{head}/{name}_sup"] = int(sup[c])
    return metrics


def pretty_report(metrics: Dict[str, float]) -> str:
    """Человекочитаемая таблица по трём головам."""
    lines = []
    specs = {"punct": PUNCT_LABELS, "para": PARA_LABELS, "cap": CAP_LABELS}
    for head, labels in specs.items():
        lines.append(f"\n=== {head.upper()} "
                     f"(macro-F1={metrics.get(head + '_f1_macro', 0):.3f}, "
                     f"acc={metrics.get(head + '_acc', 0):.3f}) ===")
        lines.append(f"{'class':<12}{'P':>8}{'R':>8}{'F1':>8}{'support':>10}")
        for name in labels:
            lines.append(
                f"{name:<12}"
                f"{metrics.get(f'{head}/{name}_p', 0):>8.3f}"
                f"{metrics.get(f'{head}/{name}_r', 0):>8.3f}"
                f"{metrics.get(f'{head}/{name}_f1', 0):>8.3f}"
                f"{metrics.get(f'{head}/{name}_sup', 0):>10}"
            )
    return "\n".join(lines)
