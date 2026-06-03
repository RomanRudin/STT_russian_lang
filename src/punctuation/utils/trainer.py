"""
trainer.py
==========
Общий цикл обучения для всех четырёх моделей.

Лосс — сумма трёх cross-entropy (punct + case + para), padding и продолжения
subword игнорируются через IGNORE_INDEX. Веса голов настраиваются.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils.labels import IGNORE_INDEX
from utils.metrics import compute_all_metrics


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
        logits = outputs[head]
        target = batch[head]
        loss = loss + w * ce(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )
    return loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    weights=(1.0, 0.5, 0.5),
) -> Dict[str, float]:
    """Вычисляет средний loss и метрики на валидационном датасете."""
    model.eval()
    total_loss = 0.0
    all_metrics = {}
    num_batches = 0
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        loss = compute_loss(outputs, batch, weights=weights)
        total_loss += loss.item()
        
        metrics = compute_all_metrics(outputs, batch)
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        
        num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    avg_metrics = {k: v / max(1, num_batches) for k, v in all_metrics.items()}
    avg_metrics["loss"] = avg_loss
    model.train()
    return avg_metrics


def train_model(
    model: nn.Module,
    dataset,
    collate_fn,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 3e-4,
    weights=(1.0, 0.5, 0.5),
    device: Optional[str] = None,
    verbose: bool = True,
    val_split: float = 0.1,
    patience: int = 3,
    save_path: Optional[str] = None,
    save_best_metric: str = "punct_accuracy",
    maximize_metric: bool = True,
) -> Dict:
    """
    Универсальный тренер с валидацией, early stopping и сохранением.
    
    Возвращает словарь с историей обучения и валидации.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Разделение на train/val
    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    if val_len > 0:
        train_ds, val_ds = random_split(dataset, [train_len, val_len])
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
    else:
        val_loader = None
        train_ds = dataset
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": [],
    }
    
    best_score = -float("inf") if maximize_metric else float("inf")
    best_epoch = -1
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
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
            total_train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_train_loss / max(1, n_batches)
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, weights=weights)
            history["val_loss"].append(val_metrics["loss"])
            history["val_metrics"].append(val_metrics)
            
            current_score = val_metrics.get(save_best_metric, val_metrics["loss"])
            if maximize_metric:
                is_better = current_score > best_score
            else:
                is_better = current_score < best_score
            
            if is_better:
                best_score = current_score
                best_epoch = epoch
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    if verbose:
                        print(f"  New best model saved with {save_best_metric}={best_score:.4f}")
            else:
                patience_counter += 1
            
            if verbose:
                print(f"  epoch {epoch:2d}/{epochs}  train_loss={avg_train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
                      f"punct_acc={val_metrics.get('punct_accuracy', 0):.3f}  case_acc={val_metrics.get('case_accuracy', 0):.3f}")
        else:
            if verbose:
                print(f"  epoch {epoch:2d}/{epochs}  train_loss={avg_train_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}, best {save_best_metric}={best_score:.4f} at epoch {best_epoch}")
            break
    
    # Загружаем лучшую модель, если есть
    if save_path and val_loader is not None and best_epoch != -1:
        model.load_state_dict(torch.load(save_path))
        if verbose:
            print(f"Loaded best model from {save_path}")
    
    return history
