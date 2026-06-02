import os
import glob
import logging
from typing import Dict, Any, List, Callable, Optional
import torch
from transformers import (
    Seq2SeqTrainer as HFTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)
from ..trainer import BaseTrainer
from acoustic.utils.config import save_config_snapshot

logger = logging.getLogger(__name__)


class WhisperTrainer(BaseTrainer):
    """Trainer tailored for Whisper models. Implements all abstract methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_training_args()
        self._setup_trainer()

    def _setup_training_args(self) -> None:
        t_cfg = self.cfg['training']
        output_dir = t_cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        save_config_snapshot(self.cfg, output_dir)

        self.training_args = Seq2SeqTrainingArguments(
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
            predict_with_generate=t_cfg['predict_with_generate'],
            generation_max_length=t_cfg['generation_max_length'],
            load_best_model_at_end=t_cfg['load_best_model_at_end'],
            metric_for_best_model=t_cfg['metric_for_best_model'],
            greater_is_better=t_cfg['greater_is_better'],
            save_total_limit=t_cfg['save_total_limit'],
            dataloader_num_workers=t_cfg.get('dataloader_num_workers', 0),
            remove_unused_columns=False,
            report_to=["tensorboard"],
            logging_dir=os.path.join(output_dir, "logs"),   # tensorboard logs
            disable_tqdm=True,
        )

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        pred_ids, label_ids = eval_pred
        preds = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids_tensor = torch.tensor(label_ids)
        label_ids_tensor = torch.where(
            label_ids_tensor != -100,
            label_ids_tensor,
            self.processor.tokenizer.pad_token_id
        )
        refs = self.processor.batch_decode(label_ids_tensor, skip_special_tokens=True)

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
        if self.data_collator is None:
            self.data_collator = DataCollatorForSeq2Seq(
                self.processor.tokenizer,
                model=self.model,
                padding=True,
            )

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
            tokenizer=self.processor.tokenizer,
            callbacks=hf_callbacks,
        )

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            return None
        def step_from_path(p):
            try:
                return int(p.split("checkpoint-")[-1])
            except:
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

        # Save final model and processor
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

    # Ensure processor is saved in every checkpoint
    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        checkpoint_folder = f"checkpoint-{self.hf_trainer.state.global_step}"
        output_dir = os.path.join(self.training_args.output_dir, checkpoint_folder)
        self.processor.save_pretrained(output_dir)
        logger.info(f"Processor saved to {output_dir}")