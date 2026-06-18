"""
visualization.py
================
Функции для визуализации метрик и сравнения моделей.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def plot_losses(
    histories: Dict[str, Dict],
    title: str = "Сравнение потерь",
    save_path: Optional[str] = None
) -> None:
    """
    Строит график train и val loss для нескольких моделей.
    histories: словарь {имя_модели: история_обучения}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label=f"{name} train", marker='o', markersize=3)
        if "val_loss" in hist and hist["val_loss"]:
            ax.plot(epochs, hist["val_loss"], label=f"{name} val", linestyle='--', marker='s', markersize=3)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_metric(
    histories: Dict[str, Dict],
    metric_name: str = "punct_accuracy",
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Строит график выбранной метрики (из val_metrics) для всех моделей.
    metric_name: ключ в val_metrics, например "punct_accuracy", "case_accuracy", "punct_f1_macro"
    """
    if title is None:
        title = f"{metric_name.replace('_', ' ').title()} на валидации"
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in histories.items():
        if "val_metrics" in hist and hist["val_metrics"]:
            epochs = range(1, len(hist["val_metrics"]) + 1)
            values = [m.get(metric_name, 0) for m in hist["val_metrics"]]
            ax.plot(epochs, values, label=name, marker='o', markersize=3)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def build_metrics_table(
    histories: Dict[str, Dict],
    metric_keys: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Строит DataFrame с лучшими значениями метрик для каждой модели.
    metric_keys: список ключей для извлечения (по умолчанию все доступные)
    """
    if metric_keys is None:
        # базовый набор
        metric_keys = ["punct_accuracy", "punct_f1_macro", "case_accuracy", "case_f1_macro", "para_accuracy"]
    rows = []
    for name, hist in histories.items():
        if "val_metrics" not in hist or not hist["val_metrics"]:
            continue
        # находим эпоху с максимальной punct_accuracy (или первой, если нет)
        if "punct_accuracy" in hist["val_metrics"][0]:
            best_idx = np.argmax([m.get("punct_accuracy", 0) for m in hist["val_metrics"]])
        else:
            best_idx = 0
        best = hist["val_metrics"][best_idx]
        row = {"Модель": name, "Лучшая эпоха": best_idx + 1}
        for key in metric_keys:
            row[key] = best.get(key, None)
        rows.append(row)
    df = pd.DataFrame(rows)
    # форматирование чисел
    for col in df.columns:
        if col not in ["Модель", "Лучшая эпоха"]:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    return df

