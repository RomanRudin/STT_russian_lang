import os
import glob
import logging
from typing import Dict, Any, Optional
import torch
from transformers import (
    Seq2SeqTrainer as HFTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from ..trainer import BaseTrainer
from acoustic.utils.config import save_config_snapshot
from acoustic.dataset.filters.text_normalizer import normalise_text

logger = logging.getLogger(__name__)


class RuT5Trainer(BaseTrainer):
    """Trainer tailored for ruT5 text correction models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = kwargs.get('metrics', [])
        self._setup_training_args()
        self._setup_trainer()

    def _setup_training_args(self) -> None:
        t_cfg = self.cfg['training']
        output_dir = t_cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        save_config_snapshot(self.cfg, output_dir)

        self.training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=t_cfg.get('batch_size', 16),
            per_device_eval_batch_size=t_cfg.get('batch_size', 16),
            learning_rate=t_cfg.get('learning_rate', 3e-4),
            num_train_epochs=t_cfg.get('num_train_epochs', 5),
            eval_strategy=t_cfg.get('eval_strategy', 'epoch'),
            save_strategy=t_cfg.get('save_strategy', 'epoch'),
            logging_strategy=t_cfg.get('logging_strategy', 'epoch'),
            fp16=False,
            predict_with_generate=True,
            load_best_model_at_end=t_cfg.get('load_best_model_at_end', True),
            metric_for_best_model=t_cfg.get('metric_for_best_model', 'wer'),
            greater_is_better=t_cfg.get('greater_is_better', False),
            save_total_limit=t_cfg.get('save_total_limit', 3),
            dataloader_num_workers=t_cfg.get('dataloader_num_workers', 0),
            dataloader_pin_memory=t_cfg.get('dataloader_pin_memory', True),
            remove_unused_columns=False,
            report_to=["tensorboard"],
            logging_dir=os.path.join(output_dir, "logs"),
            disable_tqdm=True,
        )

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        if not self.metrics:
            return {}

        pred_ids, label_ids = eval_pred
        preds = self.processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids_tensor = torch.tensor(label_ids)
        label_ids_tensor = torch.where(
            label_ids_tensor != -100,
            label_ids_tensor,
            self.processor.pad_token_id
        )
        refs = self.processor.batch_decode(label_ids_tensor, skip_special_tokens=True)

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
            ))
        hf_callbacks.extend(self.callbacks)

        self.hf_trainer = HFTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            processing_class=self.processor,
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
        latest = self._find_latest_checkpoint(self.training_args.output_dir)
        if latest:
            logger.info("Auto-resuming from latest checkpoint: %s", latest)
            self.hf_trainer.train(resume_from_checkpoint=latest)
        else:
            logger.info("Starting training from scratch")
            self.hf_trainer.train()

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