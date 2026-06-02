"""
trainer.py
==========
Общий цикл обучения для всех четырёх моделей.

Лосс — сумма трёх cross-entropy (punct + case + para), padding и продолжения
subword игнорируются через IGNORE_INDEX. Веса голов настраиваются.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.labels import IGNORE_INDEX


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights=(1.0, 0.5, 0.5),
) -> torch.Tensor:
    """Сумма cross-entropy по трём головам с весами (punct, case, para)."""
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    w_punct, w_case, w_para = weights
    loss = 0.0
    for head, w in (("punct", w_punct), ("case", w_case), ("para", w_para)):
        logits = outputs[head]                       # (B, T, C)
        target = batch[head]                         # (B, T)
        loss = loss + w * ce(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )
    return loss


def train_model(
    model,
    dataset,
    collate_fn,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 3e-4,
    weights=(1.0, 0.5, 0.5),
    device: str | None = None,
    verbose: bool = True,
) -> list:
    """Универсальный тренер. Возвращает список средних лоссов по эпохам."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            loss = compute_loss(outputs, batch, weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item()
            n += 1
        avg = total / max(1, n)
        history.append(avg)
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs}  loss = {avg:.4f}")
    return history
