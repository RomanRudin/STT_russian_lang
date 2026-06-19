import os
import glob
import logging
import numpy as np
from typing import Dict, Any, Optional
import torch
from transformers import Trainer as HFTrainer, TrainingArguments, EarlyStoppingCallback
from ..trainer import BaseTrainer
from acoustic.utils.config import save_config_snapshot

logger = logging.getLogger(__name__)

class CustomSTTTrainer(BaseTrainer):
    """Trainer tailored for custom STT models using CTC."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.processor 
        self._setup_training_args()
        self._setup_trainer()

    def _setup_training_args(self) -> None:
        t_cfg = self.cfg['training']
        output_dir = t_cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        save_config_snapshot(self.cfg, output_dir)

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=t_cfg['per_device_train_batch_size'],
            per_device_eval_batch_size=t_cfg['per_device_eval_batch_size'],
            gradient_accumulation_steps=t_cfg['gradient_accumulation_steps'],
            learning_rate=t_cfg['learning_rate'],
            warmup_steps=t_cfg['warmup_steps'],
            num_train_epochs=t_cfg['num_train_epochs'],
            eval_strategy=t_cfg.get('eval_strategy', 'epoch'),
            save_strategy=t_cfg.get('save_strategy', 'epoch'),
            logging_strategy=t_cfg.get('logging_strategy', 'epoch'),
            fp16=t_cfg['fp16'] and torch.cuda.is_available(),
            load_best_model_at_end=t_cfg['load_best_model_at_end'],
            metric_for_best_model=t_cfg['metric_for_best_model'],
            greater_is_better=t_cfg['greater_is_better'],
            save_total_limit=t_cfg['save_total_limit'],
            dataloader_num_workers=t_cfg.get('dataloader_num_workers', 4),
            remove_unused_columns=False,
            report_to=["tensorboard"],
            logging_dir=os.path.join(output_dir, "logs"),
            disable_tqdm=True,
        )

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        logits, label_ids = eval_pred
        pred_ids = np.argmax(logits, axis=-1)

        def decode_ctc(preds_seq, blank_id=0):
            decoded = []
            for pred in preds_seq:
                chars = []
                prev_char = -1
                for char_id in pred:
                    if char_id != prev_char and char_id != blank_id:
                        chars.append(char_id)
                    prev_char = char_id
                decoded.append(self.tokenizer.decode(chars))
            return decoded

        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        
        preds = decode_ctc(pred_ids, blank_id=0)
        refs = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        from acoustic.dataset.filters.text_normalizer import normalise_text
        preds_norm = [normalise_text(p) for p in preds]
        refs_norm = [normalise_text(r) for r in refs]

        pairs = [(p, r) for p, r in zip(preds_norm, refs_norm) if r.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        
        preds_clean, refs_clean = zip(*pairs)

        results = {}
        for metric_fn in self.metrics:
            results.update(metric_fn({"predictions": list(preds_clean), "references": list(refs_clean)}))
        return results

    def _setup_trainer(self) -> None:
        hf_callbacks = []
        if self.cfg['training'].get('early_stopping_patience'):
            hf_callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.cfg['training']['early_stopping_patience'],
                early_stopping_threshold=self.cfg['training'].get('early_stopping_threshold', 0.001),
            ))
        hf_callbacks.extend(self.callbacks)

        self.hf_trainer = HFTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=hf_callbacks,
        )

    def train(self) -> None:
        latest = self._find_latest_checkpoint(self.training_args.output_dir)
        if latest:
            logger.info("Auto-resuming from latest checkpoint: %s", latest)
            self.hf_trainer.train(resume_from_checkpoint=latest)
        else:
            logger.info("Starting training from scratch")
            self.hf_trainer.train()

        final_dir = os.path.join(self.training_args.output_dir, "final_model")
        self.hf_trainer.save_model(final_dir)
        save_config_snapshot(self.cfg, final_dir)

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints: return None
        return max(checkpoints, key=lambda p: int(p.split("checkpoint-")[-1]) if p.split("checkpoint-")[-1].isdigit() else 0)