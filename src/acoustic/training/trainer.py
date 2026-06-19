import logging
from typing import Dict, Any, List, Callable, Optional
import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Abstract trainer interface.
    Concrete implementations (e.g., WhisperTrainer) must override
    every method that raises NotImplementedError.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: torch.nn.Module,
        processor: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any],
        metrics: List[Callable],
        callbacks: List[TrainerCallback],
        data_collator: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics = metrics
        self.callbacks = callbacks
        self.data_collator = data_collator
        # Subclasses must call _setup_training_args() and _setup_trainer() in their __init__

    def _setup_training_args(self) -> None:
        """Create Seq2SeqTrainingArguments. Must be implemented by subclass."""
        raise NotImplementedError

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute metrics from predictions and labels. Must be implemented."""
        raise NotImplementedError

    def _setup_trainer(self) -> None:
        """Instantiate the underlying HuggingFace Trainer. Must be implemented."""
        raise NotImplementedError

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """Find the latest checkpoint in a directory. Must be implemented."""
        raise NotImplementedError

    def train(self) -> None:
        """Run training (possibly resuming). Must be implemented."""
        raise NotImplementedError

    def evaluate(self, test_dataset: Optional[Any] = None) -> Dict[str, float]:
        """Run evaluation on the given dataset. Must be implemented."""
        raise NotImplementedError