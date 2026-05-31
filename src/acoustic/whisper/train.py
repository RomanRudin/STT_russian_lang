"""
    python -m src.acoustic.whisper.train                        # full fine-tune
    python -m src.acoustic.whisper.train --lora                 # LoRA (~8 GB VRAM)
    python -m src.acoustic.whisper.train --config configs/acoustic.yaml
    python -m src.acoustic.whisper.train --resume_from ./checkpoints/whisper-ru/checkpoint-3000
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Optional

import yaml
import torch
from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.acoustic.whisper.dataset import DataCollatorSpeechSeq2SeqWithPadding, build_dataset
from src.acoustic.whisper.evaluate import make_compute_metrics
from src.acoustic.whisper.model import build_model

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def _install_tqdm_logging() -> None:
    root = logging.getLogger()
    for h in root.handlers[:]:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, TqdmLoggingHandler):
            root.removeHandler(h)
    handler = TqdmLoggingHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                          datefmt="%H:%M:%S")
    )
    root.addHandler(handler)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TqdmTrainingCallback(TrainerCallback):

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self._train_bar: Optional[tqdm] = None
        self._eval_bar:  Optional[tqdm] = None
        self._step_t0:   float = 0.0

    def on_train_begin(
        self, args, state: TrainerState, control: TrainerControl, **kw
    ) -> None:
        self._train_bar = tqdm(
            total=self.total_steps,
            initial=state.global_step,   # non-zero when resuming
            desc="Training ",
            unit="step",
            dynamic_ncols=True,
            colour="green",
            position=0,
            leave=True,
        )

    def on_train_end(
        self, args, state: TrainerState, control: TrainerControl, **kw
    ) -> None:
        if self._train_bar:
            self._train_bar.close()
            self._train_bar = None

    def on_step_begin(
        self, args, state: TrainerState, control: TrainerControl, **kw
    ) -> None:
        self._step_t0 = time.perf_counter()

    def on_step_end(
        self, args, state: TrainerState, control: TrainerControl, **kw
    ) -> None:
        if self._train_bar is None:
            return

        elapsed = time.perf_counter() - self._step_t0
        pf: Dict = {"step": state.global_step}

        # Pull the most recently logged values from history
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                pf["loss"] = f"{last['loss']:.4f}"
            if "learning_rate" in last:
                pf["lr"] = f"{last['learning_rate']:.2e}"

        pf["s/step"] = f"{elapsed:.2f}"
        self._train_bar.set_postfix(pf)
        self._train_bar.update(1)

    def on_log(
        self, args, state: TrainerState, control: TrainerControl,
        logs: Optional[Dict] = None, **kw
    ) -> None:
        if not logs or self._train_bar is None:
            return
        parts = [f"step {state.global_step:>6}"]
        for key in ("loss", "eval_loss", "eval_wer", "eval_cer", "learning_rate"):
            if key in logs:
                v = logs[key]
                if isinstance(v, float):
                    fmt = f"{v:.2e}" if "lr" in key else f"{v:.4f}"
                    parts.append(f"{key}={fmt}")
        tqdm.write("  " + "  |  ".join(parts))

    def on_save(
        self, args, state: TrainerState, control: TrainerControl, **kw
    ) -> None:
        tqdm.write(f"  \u2713 checkpoint saved  (step {state.global_step})")


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, **kw
    ) -> None:
        if self._eval_bar:
            self._eval_bar.close()
        self._eval_bar = tqdm(
            desc="Evaluating",
            unit="batch",
            dynamic_ncols=True,
            colour="blue",
            position=1,
            leave=False,
        )

    def on_prediction_step(
        self, args, state: TrainerState, control: TrainerControl, **kw
    ) -> None:
        if self._eval_bar is not None:
            self._eval_bar.update(1)

    # HuggingFace does not expose on_evaluate_end as a named hook yet,
    # so we close the bar inside on_log when eval metrics appear.
    def _close_eval_bar(self) -> None:
        if self._eval_bar:
            self._eval_bar.close()
            self._eval_bar = None



def _banner(title: str) -> None:
    line = "\u2500" * 62
    tqdm.write(f"\n{line}\n  {title}\n{line}")


def _print_table(rows: list[tuple], headers: tuple) -> None:
    col_w = [max(len(str(headers[i])),
                 max(len(str(r[i])) for r in rows))
             for i in range(len(headers))]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    tqdm.write(fmt.format(*headers))
    tqdm.write("  " + "  ".join("\u2500" * w for w in col_w))
    for row in rows:
        tqdm.write(fmt.format(*row))


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Whisper for Russian STT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       default="configs/acoustic.yaml")
    p.add_argument("--experiment",   default=None,
                   help=(
                       "Short name for this run, e.g. 'medium_lora' or 'large_crowd_only'. "
                       "Sets output_dir to checkpoints/<experiment> and saves a config "
                       "snapshot alongside the checkpoint so every run is reproducible. "
                       "If omitted, output_dir from the YAML config is used."
                   ))
    p.add_argument("--lora",         action="store_true",
                   help="LoRA adapters — saves ~30 GB VRAM vs full fine-tune")
    p.add_argument("--resume_from",  default=None,
                   help="Path to a checkpoint directory to resume from")
    p.add_argument("--push_to_hub",  action="store_true",
                   help="Push best model to HuggingFace Hub after training")
    p.add_argument("--hub_model_id", default=None,
                   help="Hub repo, e.g. 'you/whisper-large-v3-russian'")
    p.add_argument("--set",          nargs="+", default=[], metavar="KEY=VALUE",
                   help=(
                       "Override any scalar config value without editing the YAML. "
                       "Dot-notation for nested keys. Examples:\n"
                       "  --set training.learning_rate=5e-6\n"
                       "  --set model.name=openai/whisper-medium\n"
                       "  --set data.golos_configs=[crowd]"
                   ))
    return p.parse_args()


def _apply_overrides(cfg: dict, overrides: list) -> dict:
    """
    Apply --set KEY=VALUE overrides to a nested config dict.
    KEY uses dot-notation: training.learning_rate=5e-6
    Scalars (float/int/bool/str) and [list] literals are all supported.
    """
    import ast
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set argument must be KEY=VALUE, got: {item!r}")
        key, _, raw = item.partition("=")
        keys = key.strip().split(".")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
        tqdm.write(f"  override: {key} = {value!r}")
    return cfg

def main() -> None:
    _install_tqdm_logging()
    args = parse_args()

    # Config
    _banner("1 / 5   Config")
    with open(args.config, encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)

    t_cfg      = cfg["training"]
    output_dir = t_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Apply --set KEY=VALUE overrides before anything reads cfg
    if args.set:
        cfg = _apply_overrides(cfg, args.set)

    # --experiment gives the run a name and its own output directory
    if args.experiment:
        base = cfg["training"].get("experiments_dir", "checkpoints")
        cfg["training"]["output_dir"] = os.path.join(base, args.experiment)
        cfg["_experiment"] = args.experiment
        # Recompute t_cfg and output_dir with the new path
        t_cfg      = cfg["training"]
        output_dir = t_cfg["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

    # Save an exact config snapshot next to the checkpoint
    import yaml as _yaml
    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w", encoding="utf-8") as _fh:
        _yaml.dump(cfg, _fh, allow_unicode=True, default_flow_style=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _print_table(
        rows=[
            ("Model",      cfg["model"]["name"]),
            ("LoRA",       str(args.lora)),
            ("Max steps",  str(t_cfg["max_steps"])),
            ("Batch size", f"{t_cfg['per_device_train_batch_size']} × "
                           f"{t_cfg['gradient_accumulation_steps']} accumulation"),
            ("Device",     device),
            ("Output",     output_dir),
        ],
        headers=("Setting", "Value"),
    )

    # Model
    _banner("2 / 5   Model")
    with auto_tqdm(total=1, desc="Loading model", unit="model",
                   dynamic_ncols=True, colour="cyan") as pbar:
        model, processor = build_model(cfg, use_lora=args.lora)
        pbar.update(1)

    # Dataset
    _banner("3 / 5   Dataset  (streams + caches on first run)")
    dataset = build_dataset(cfg, processor.feature_extractor, processor.tokenizer)

    _print_table(
        rows=[(split, f"{len(ds):,}") for split, ds in dataset.items()],
        headers=("Split", "Samples"),
    )

    collator        = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = make_compute_metrics(processor.tokenizer)

    # Training
    _banner("4 / 5   Training")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        warmup_steps=t_cfg["warmup_steps"],
        max_steps=t_cfg["max_steps"],
        eval_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=t_cfg["save_steps"],
        logging_steps=t_cfg["logging_steps"],
        fp16=t_cfg["fp16"] and torch.cuda.is_available(),
        predict_with_generate=t_cfg["predict_with_generate"],
        generation_max_length=t_cfg["generation_max_length"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=t_cfg["greater_is_better"],
        save_total_limit=t_cfg["save_total_limit"],
        dataloader_num_workers=t_cfg["dataloader_num_workers"],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        disable_tqdm=True,   # we supply our own bars via TqdmTrainingCallback
    )

    tqdm_cb = TqdmTrainingCallback(total_steps=t_cfg["max_steps"])

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[
            tqdm_cb,
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001,
            ),
        ],
    )

    t0           = time.perf_counter()
    train_result = trainer.train()#resume_from_checkpoint=args.resume_from)
    elapsed_h    = (time.perf_counter() - t0) / 3600
    tqdm.write(f"\n  Training finished in {elapsed_h:.2f} h")

    # Sa
    with auto_tqdm(total=2, desc="Saving    ", unit="item",
                   dynamic_ncols=True, colour="cyan") as pbar:
        trainer.save_model(output_dir)
        pbar.set_postfix({"saved": "model"})
        pbar.update(1)
        processor.save_pretrained(output_dir)
        pbar.set_postfix({"saved": "model + processor"})
        pbar.update(1)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Final test evaluation
    _banner("5 / 5   Final evaluation on test split")

    if len(dataset["test"]) == 0:
        tqdm.write("  Test split is empty — skipping.")
    else:
        test_metrics = trainer.evaluate(
            eval_dataset=dataset["test"],
            metric_key_prefix="test",
        )
        tqdm_cb._close_eval_bar()

        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

        _print_table(
            rows=[
                ("WER",  f"{test_metrics.get('test_wer', float('nan')):.4f}"),
                ("CER",  f"{test_metrics.get('test_cer', float('nan')):.4f}"),
                ("Loss", f"{test_metrics.get('test_loss', float('nan')):.4f}"),
            ],
            headers=("Metric", "Value"),
        )

    # optional
    if args.push_to_hub:
        with auto_tqdm(total=1, desc="Hub push", unit="model",
                       dynamic_ncols=True, colour="magenta") as pbar:
            trainer.push_to_hub()
            pbar.update(1)

    tqdm.write(f"\n  \u2713 Done.  Model saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
