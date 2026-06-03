import os
import glob
import logging
import numpy as np
from typing import Dict, Any, Optional

import torch
from transformers import (
    Trainer as HFTrainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from ..trainer import BaseTrainer
from acoustic.utils.config import save_config_snapshot

logger = logging.getLogger(__name__)

class Wav2Vec2Trainer(BaseTrainer):
    """Trainer tailored for Wav2Vec2 (CTC) models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_training_args()
        self._setup_trainer()

    def _setup_training_args(self) -> None:
        t_cfg = self.cfg['training']
        output_dir = t_cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        save_config_snapshot(self.cfg, output_dir)

        # Используем обычные TrainingArguments, так как это не генеративная модель
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=t_cfg['per_device_train_batch_size'],
            per_device_eval_batch_size=t_cfg['per_device_eval_batch_size'],
            gradient_accumulation_steps=t_cfg['gradient_accumulation_steps'],
            learning_rate=t_cfg['learning_rate'],
            warmup_steps=t_cfg['warmup_steps'],
            num_train_epochs=t_cfg['num_train_epochs'],
            evaluation_strategy=t_cfg.get('eval_strategy', 'epoch'),
            save_strategy=t_cfg.get('save_strategy', 'epoch'),
            logging_strategy=t_cfg.get('logging_strategy', 'epoch'),
            fp16=t_cfg['fp16'] and torch.cuda.is_available(),
            load_best_model_at_end=t_cfg['load_best_model_at_end'],
            metric_for_best_model=t_cfg['metric_for_best_model'],
            greater_is_better=t_cfg['greater_is_better'],
            save_total_limit=t_cfg['save_total_limit'],
            dataloader_num_workers=t_cfg.get('dataloader_num_workers', 0),
            remove_unused_columns=False,
            report_to=["tensorboard"],
            logging_dir=os.path.join(output_dir, "logs"),
            disable_tqdm=True,
        )

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        logits, label_ids = eval_pred

        # Для CTC берем токен с максимальной вероятностью (жадный декодинг)
        pred_ids = np.argmax(logits, axis=-1)

        # Заменяем -100 на pad_token_id, чтобы не сломать декодирование
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Декодируем токены в текст
        preds = self.processor.batch_decode(pred_ids)
        # Важно: group_tokens=False для лейблов, так как они не проходят через CTC-сжатие
        refs = self.processor.batch_decode(label_ids, group_tokens=False)

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
            tokenizer=self.processor.feature_extractor, # Для Wav2Vec2 Trainer иногда требует feature_extractor
            callbacks=hf_callbacks,
        )

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            return None
        def step_from_path(p):
            try:
                return int(p.split("checkpoint-")[-1])
            except ValueError:
                return 0
        return max(checkpoints, key=step_from_path)

    def train(self) -> None:
        resume_from = self.cfg['training'].get('resume_from')
        if resume_from:
            logger.info("Resuming from explicit checkpoint: %s", resume_from)
            self.hf_trainer.train(resume_from_checkpoint=resume_from)
        else:
            latest = self._find_latest_checkpoint(self.training_args.output_dir)
            if latest:
                logger.info("Auto-resuming from latest checkpoint: %s", latest)
                self.hf_trainer.train(resume_from_checkpoint=latest)
            else:
                logger.info("Starting training from scratch")
                self.hf_trainer.train()

        # Сохранение итоговой модели
        final_dir = os.path.join(self.training_args.output_dir, "final_model")
        self.hf_trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        save_config_snapshot(self.cfg, final_dir)
        logger.info("Final model saved to %s", final_dir)

    def evaluate(self, test_dataset: Optional[Any] = None) -> Dict[str, float]:
        if test_dataset is None:
            test_dataset = getattr(self, 'eval_dataset', None)
        if test_dataset is None:
            raise ValueError("No evaluation dataset provided")
        return self.hf_trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

# Регистрируем тренера