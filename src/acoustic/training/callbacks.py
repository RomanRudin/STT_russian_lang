"""
acoustic/training/callbacks.py
Callback classes, registry, and convenience getter.
"""

from typing import Optional
from transformers import TrainerCallback
from tqdm import tqdm
from acoustic.utils.registry import PluginRegistry


class TqdmCallback(TrainerCallback):
    """Displays a tqdm progress bar for training steps."""

    def __init__(self, total_steps: Optional[int] = None):
        self.total_steps = total_steps
        self.pbar: Optional[tqdm] = None

    def on_train_begin(self, args, state, control, **kwargs):
        total = self.total_steps if self.total_steps is not None else state.max_steps
        self.pbar = tqdm(
            total=total,
            initial=state.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar and state.global_step > self.pbar.n:
            self.pbar.update(state.global_step - self.pbar.n)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()


class EvalTqdmCallback(TrainerCallback):
    """Optional callback to show a progress bar for evaluation."""

    def __init__(self):
        self.eval_pbar: Optional[tqdm] = None

    def on_evaluate(self, args, state, control, **kwargs):
        eval_dataloader = kwargs.get("eval_dataloader")
        if eval_dataloader is not None:
            total = len(eval_dataloader)
            self.eval_pbar = tqdm(
                total=total, desc="Evaluating", unit="batch",
                dynamic_ncols=True, leave=False
            )

    def on_prediction_step(self, args, state, control, **kwargs):
        if self.eval_pbar:
            self.eval_pbar.update(1)

    def on_evaluate_end(self, args, state, control, **kwargs):
        if self.eval_pbar:
            self.eval_pbar.close()


# Registry: stores class constructors, not instances
CALLBACKS = PluginRegistry("callback")

# Manual registration
CALLBACKS.register("tqdm", TqdmCallback)
CALLBACKS.register("eval_tqdm", EvalTqdmCallback)


def get_callback(name: str, **kwargs) -> TrainerCallback:
    """
    Instantiate a callback by its registered name.
    Additional keyword arguments are passed to the constructor.
    """
    cb_class = CALLBACKS.get(name)
    return cb_class(**kwargs)