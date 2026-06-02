"""
train.py
========
Единый цикл обучения для LSTM / Transformer / RuBERT. Все модели возвращают
один и тот же словарь логитов {punct, para, cap}, поэтому loss и шаг обучения
общие.

Суммарный лосс = w_punct * CE(punct) + w_para * CE(para) + w_cap * CE(cap),
где IGNORE_INDEX исключает паддинг и субтокены-продолжения. Для пунктуации
используются веса классов (класс O преобладает).
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainConfig, IGNORE_INDEX


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA недоступна — перехожу на CPU.")
        return torch.device("cpu")
    return torch.device(name)


def build_loss(cfg: TrainConfig, device: torch.device):
    """Возвращает три CrossEntropy (с весами классов для пунктуации)."""
    w = torch.tensor(cfg.punct_class_weights, dtype=torch.float32, device=device)
    return {
        "punct": nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE_INDEX),
        "para": nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
        "cap": nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
    }


def compute_loss(logits: Dict[str, torch.Tensor], batch: Dict,
                 loss_fns: Dict, cfg: TrainConfig) -> torch.Tensor:
    def ce(name):
        lg = logits[name].reshape(-1, logits[name].size(-1))
        tg = batch[name].reshape(-1)
        return loss_fns[name](lg, tg)

    return (cfg.w_punct * ce("punct")
            + cfg.w_para * ce("para")
            + cfg.w_cap * ce("cap"))


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    val_loader: Optional[DataLoader] = None,
    is_pretrained: bool = False,
    eval_fn: Optional[Callable] = None,
) -> nn.Module:
    """
    Обучает модель `cfg.epochs` эпох. Для предобученных моделей задаёт
    раздельные lr (энкодер vs головы) и линейный warmup.
    Возвращает обученную модель (лучшую по val, если задан eval_fn+val_loader).
    """
    set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    model.to(device)
    loss_fns = build_loss(cfg, device)

    # Оптимизатор
    if is_pretrained and hasattr(model, "param_groups"):
        groups = model.param_groups(encoder_lr=2e-5, head_lr=1e-3)
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=cfg.weight_decay)

    total_steps = max(len(train_loader) * cfg.epochs, 1)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_score, best_state = -1.0, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            logits = model(**batch)
            loss = compute_loss(logits, batch, loss_fns, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        avg = running / max(len(train_loader), 1)
        msg = f"[epoch {epoch}/{cfg.epochs}] train_loss={avg:.4f}"

        if val_loader is not None and eval_fn is not None:
            metrics = eval_fn(model, val_loader, device)
            score = metrics.get("punct_f1_macro", 0.0)
            msg += f"  val_punct_F1={score:.4f}"
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[train] загружены лучшие веса (val_punct_F1={best_score:.4f}).")
    return model
