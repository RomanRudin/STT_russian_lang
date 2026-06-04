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

from .config import TrainConfig, IGNORE_INDEX, NUM_PUNCT, NUM_PARA, NUM_CAP


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA недоступна — перехожу на CPU.")
        return torch.device("cpu")
    return torch.device(name)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) для token-classification с дисбалансом.

    CE штрафует все ошибки одинаково, и модель «залипает» на доминирующем
    классе O. Focal домножает потерю на (1 - p_t)^gamma: легко угаданные
    примеры (уверенный O) почти не дают градиента, а трудные редкие знаки —
    дают. Это обычно лучший рычаг против перекоса классов на нашей задаче.

    weight   : веса классов (как в CrossEntropy), опционально.
    gamma    : сила фокусировки (0 -> обычный взвешенный CE).
    """

    def __init__(self, weight: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C), target: (N,)
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        ce = torch.nn.functional.nll_loss(
            logp, target, weight=self.weight,
            ignore_index=self.ignore_index, reduction="none",
        )
        # p_t = вероятность истинного класса
        valid = target != self.ignore_index
        pt = torch.zeros_like(ce)
        if valid.any():
            idx = target[valid].clamp_min(0)
            pt[valid] = logp[valid].gather(1, idx.unsqueeze(1)).squeeze(1).exp()
        focal = ((1 - pt) ** self.gamma) * ce
        return focal[valid].mean() if valid.any() else focal.sum() * 0.0


def compute_class_weights(examples, head: str, num_classes: int,
                          cap: float = 10.0) -> torch.Tensor:
    """
    Автоматические веса классов по обратной частоте в train.

    head : 'punct' | 'para' | 'cap' — какое поле Example агрегировать.
    Возвращает тензор весов длины num_classes, нормированный к среднему 1,
    с обрезкой сверху (cap), чтобы сверхредкие классы не взрывали лосс.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    attr = {"punct": "punct_ids", "para": "para_ids", "cap": "cap_ids"}[head]
    for ex in examples:
        for lab in getattr(ex, attr):
            if 0 <= lab < num_classes:
                counts[lab] += 1
    counts = np.maximum(counts, 1.0)             # избегаем деления на ноль
    inv = counts.sum() / counts                  # обратная частота
    inv = np.sqrt(inv)                           # сглаживание (мягче, чем чистая 1/freq)
    inv = inv / inv.mean()                       # нормировка к среднему 1
    inv = np.clip(inv, 0.3, cap)
    return torch.tensor(inv, dtype=torch.float32)


def build_loss(cfg: TrainConfig, device: torch.device,
               train_examples=None) -> Dict:
    """
    Возвращает три функции потерь (по одной на голову).

    Тип задаётся cfg.loss_type ('focal' | 'ce'). Веса классов берутся
    автоматически по частоте в train (cfg.auto_class_weights=True) либо из
    cfg.punct_class_weights для пунктуации.
    """
    # веса классов
    if cfg.auto_class_weights and train_examples is not None:
        w_punct = compute_class_weights(train_examples, "punct", NUM_PUNCT).to(device)
        w_para = compute_class_weights(train_examples, "para", NUM_PARA).to(device) if cfg.balance_para_cap else None
        w_cap = compute_class_weights(train_examples, "cap", NUM_CAP).to(device) if cfg.balance_para_cap else None
    else:
        w_punct = torch.tensor(cfg.punct_class_weights, dtype=torch.float32, device=device)
        w_para = None
        w_cap = None

    def make(weight):
        if cfg.loss_type == "focal":
            return FocalLoss(weight=weight, gamma=cfg.focal_gamma, ignore_index=IGNORE_INDEX)
        return nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)

    return {"punct": make(w_punct), "para": make(w_para), "cap": make(w_cap)}


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
    train_examples=None,
) -> nn.Module:
    """
    Обучает модель `cfg.epochs` эпох. Для предобученных моделей задаёт
    раздельные lr (энкодер vs головы) и линейный warmup.
    Возвращает обученную модель (лучшую по val, если задан eval_fn+val_loader).

    train_examples : список Example для автоподбора весов классов
                     (cfg.auto_class_weights). Можно не передавать — тогда
                     берутся ручные cfg.punct_class_weights.
    """
    set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    model.to(device)
    loss_fns = build_loss(cfg, device, train_examples=train_examples)
    print(f"[train] loss={cfg.loss_type}"
          f"{f' (gamma={cfg.focal_gamma})' if cfg.loss_type=='focal' else ''}, "
          f"auto_class_weights={cfg.auto_class_weights and train_examples is not None}")

    # Оптимизатор
    if is_pretrained and hasattr(model, "param_groups"):
        # энкодер дообучаем мягко, новые головы (+акустика) учим быстрее,
        # иначе при коротком обучении головы не успевают сойти с инициализации
        groups = model.param_groups(encoder_lr=getattr(cfg, "encoder_lr", 3e-5),
                                    head_lr=getattr(cfg, "head_lr", 3e-3))
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
