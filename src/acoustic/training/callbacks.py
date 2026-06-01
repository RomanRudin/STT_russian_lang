from typing import Dict, Any, Optional
from transformers import TrainerCallback
from tqdm import tqdm


class TqdmCallback(TrainerCallback):
    """
    A callback that displays a tqdm progress bar for training steps.
    Compatible with Seq2SeqTrainer and standard Trainer.
    """

    def __init__(self, total_steps: Optional[int] = None):
        self.total_steps = total_steps
        self.pbar: Optional[tqdm] = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize progress bar at the start of training."""
        total = self.total_steps if self.total_steps is not None else state.max_steps
        self.pbar = tqdm(
            total=total,
            initial=state.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        """Update progress bar after each training step."""
        if self.pbar and state.global_step > self.pbar.n:
            self.pbar.update(state.global_step - self.pbar.n)

    def on_train_end(self, args, state, control, **kwargs):
        """Close progress bar when training finishes."""
        if self.pbar:
            self.pbar.close()


class EvalTqdmCallback(TrainerCallback):
    """
    Optional callback to show a separate progress bar for evaluation.
    """

    def __init__(self):
        self.eval_pbar: Optional[tqdm] = None

    def on_evaluate(self, args, state, control, **kwargs):
        """Called before evaluation begins."""
        # Estimate number of evaluation batches
        eval_dataloader = kwargs.get("eval_dataloader")
        if eval_dataloader is not None:
            total = len(eval_dataloader)
            self.eval_pbar = tqdm(
                total=total, desc="Evaluating", unit="batch", dynamic_ncols=True, leave=False
            )

    def on_prediction_step(self, args, state, control, **kwargs):
        """Update evaluation progress bar after each batch."""
        if self.eval_pbar:
            self.eval_pbar.update(1)

    def on_evaluate_end(self, args, state, control, **kwargs):
        """Close evaluation progress bar."""
        if self.eval_pbar:
            self.eval_pbar.close()


# Registry for dynamic loading
CALLBACKS = {
    "tqdm": TqdmCallback,
    "eval_tqdm": EvalTqdmCallback,
}


def get_callback(name: str, **kwargs) -> TrainerCallback:
    """
    Instantiate a callback by its registered name.

    Args:
        name: Callback name (e.g., "tqdm").
        **kwargs: Additional arguments passed to the callback constructor.

    Returns:
        An instance of the requested callback.
    """
    if name not in CALLBACKS:
        raise ValueError(f"Unknown callback: {name}. Available: {list(CALLBACKS.keys())}")
    return CALLBACKS[name](**kwargs)