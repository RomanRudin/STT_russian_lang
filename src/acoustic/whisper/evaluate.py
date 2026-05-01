"""
src/acoustic/evaluate.py

Metric computation for Whisper STT fine-tuning.

compute_metrics() is passed directly to HuggingFace Trainer.
It decodes model predictions and computes WER + CER using the
`evaluate` library (backed by jiwer).
"""

import re
import unicodedata
from typing import Callable, Tuple

import numpy as np
import evaluate as hf_evaluate
from transformers import WhisperTokenizer

_PUNCT = re.compile(r"[^\u0400-\u04FF\u0030-\u0039\s]")  # keep Cyrillic + digits


def normalise_for_metric(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = _PUNCT.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_compute_metrics(
    tokenizer: WhisperTokenizer,
) -> Callable[[any], dict]:
    # Pass the returned callable to Trainer(compute_metrics=...).
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    def compute_metrics(eval_pred) -> dict:
        pred_ids, label_ids = eval_pred

        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

        # Decode predictions and references
        predictions = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        references  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalise for metric computation (strip punctuation, lower-case)
        predictions_norm = [normalise_for_metric(p) for p in predictions]
        references_norm  = [normalise_for_metric(r) for r in references]

        # Filter out empty references (can happen with very short audio)
        pairs = [
            (p, r) for p, r in zip(predictions_norm, references_norm) if r.strip()
        ]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}

        preds_clean, refs_clean = zip(*pairs)

        wer = wer_metric.compute(predictions=list(preds_clean), references=list(refs_clean))
        cer = cer_metric.compute(predictions=list(preds_clean), references=list(refs_clean))

        return {
            "wer": round(wer, 4),
            "cer": round(cer, 4),
        }

    return compute_metrics



def evaluate_model(
    model,
    dataset,
    processor,
    batch_size: int = 8,
    device: str = "cuda",
    max_samples: int = None,
) -> Tuple[float, float]:
    """
    Runs inference on `dataset` and prints per-sample errors + aggregate WER/CER.

    Args:
        model:       Fine-tuned WhisperForConditionalGeneration
        dataset:     HuggingFace Dataset with 'input_features' and 'labels'
        processor:   Matching WhisperProcessor
        batch_size:  Inference batch size
        device:      'cuda' or 'cpu'
        max_samples: Truncate dataset for quick smoke-tests

    Returns:
        (wer, cer) as floats in [0, 1]
    """
    import torch
    from torch.utils.data import DataLoader

    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    model = model.to(device).eval()

    all_preds, all_refs = [], []

    loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in loader:
            input_features = batch["input_features"].to(device)
            labels         = batch["labels"]
            labels = torch.where(labels != -100, labels, processor.tokenizer.pad_token_id)

            # Generate transcription
            predicted_ids = model.generate(input_features)

            preds = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            refs  = processor.tokenizer.batch_decode(labels,        skip_special_tokens=True)

            all_preds.extend([normalise_for_metric(p) for p in preds])
            all_refs.extend( [normalise_for_metric(r) for r in refs ])

    # Filter empty refs
    pairs = [(p, r) for p, r in zip(all_preds, all_refs) if r.strip()]
    preds_clean, refs_clean = zip(*pairs) if pairs else ([], [])

    wer = wer_metric.compute(predictions=list(preds_clean), references=list(refs_clean))
    cer = cer_metric.compute(predictions=list(preds_clean), references=list(refs_clean))

    print(f"\n{'=' * 50}")
    print(f"  WER : {wer:.4f}  ({wer*100:.2f} %)")
    print(f"  CER : {cer:.4f}  ({cer*100:.2f} %)")
    print(f"{'=' * 50}\n")

    # Print 5 random examples for a sanity check
    import random
    sample_indices = random.sample(range(len(preds_clean)), min(5, len(preds_clean)))
    for i in sample_indices:
        print(f"  REF : {refs_clean[i]}")
        print(f"  PRED: {preds_clean[i]}")
        print()

    return wer, cer
