from typing import Dict, Any, List
import torch
from transformers import PreTrainedTokenizerBase


class RuT5DataCollator:
    """Data collator for ruT5: adds task prefix, tokenizes input and target texts."""

    def __init__(
        self,
        processor: PreTrainedTokenizerBase,
        max_length: int = 128,
        task_prefix: str = "исправь: ",
        **kwargs
    ):
        self.processor = processor
        self.max_length = max_length
        self.task_prefix = task_prefix

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        inputs = [self.task_prefix + f["stt_text"] for f in features]
        targets = [f["reference"] for f in features]

        model_inputs = self.processor(
            inputs,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        labels = self.processor(
            targets,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        labels_ids = labels["input_ids"]
        labels_ids = labels_ids.masked_fill(
            labels_ids == self.processor.pad_token_id, -100
        )

        model_inputs["labels"] = labels_ids
        return model_inputs