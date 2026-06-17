"""
models/pretrained_model.py
==========================
Предобученные модели (вариант A, late fusion):
    * DeepPavlov/rubert-base-cased   — основная;
    * cointegrated/rubert-tiny2      — лёгкая (слабое железо / быстрый прогон).

Архитектура единая: HF-энкодер выдаёт контекстные эмбеддинги субтокенов,
акустика слова прокинута на его субтокены и закодирована AcousticEncoder,
затем late fusion и три головы.

Выбор модели — через PretrainedConfig.model_name (или пресет в config.py).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import PretrainedConfig
from .heads import AcousticEncoder, MultiTaskHeads, fuse


class PretrainedPunctuator(nn.Module):
    def __init__(self, cfg: PretrainedConfig):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        self.cfg = cfg
        hf_cfg = AutoConfig.from_pretrained(cfg.model_name)
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        text_dim = hf_cfg.hidden_size

        if cfg.use_acoustic:
            self.acoustic_enc = AcousticEncoder(cfg.acoustic_dim, cfg.acoustic_hidden, cfg.dropout)
            fused_dim = text_dim + self.acoustic_enc.out_dim
        else:
            self.acoustic_enc = None
            fused_dim = text_dim

        self.heads = MultiTaskHeads(fused_dim, cfg.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                acoustic: Optional[torch.Tensor] = None, **_) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_repr = out.last_hidden_state            # (B, T, hidden)

        if self.acoustic_enc is not None:
            ac = acoustic if acoustic is not None else torch.zeros(
                *input_ids.shape, self.cfg.acoustic_dim, device=input_ids.device)
            fused = fuse(text_repr, self.acoustic_enc(ac))
        else:
            fused = text_repr

        return self.heads(fused)

    # Удобно при дообучении: разные lr для энкодера и новых голов.
    def param_groups(self, encoder_lr: float, head_lr: float):
        enc_params = list(self.encoder.parameters())
        new_params = [p for n, p in self.named_parameters()
                      if not n.startswith("encoder.")]
        return [
            {"params": enc_params, "lr": encoder_lr},
            {"params": new_params, "lr": head_lr},
        ]


def load_hf_tokenizer(model_name: str):
    """Токенизатор HF для PretrainedDataset."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)
