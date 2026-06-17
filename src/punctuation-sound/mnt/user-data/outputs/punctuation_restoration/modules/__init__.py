"""
modules — пакет восстановления пунктуации (мультимодальный, late fusion / вариант A).

Публичный API для ноутбука и встраивания в SpeechToText-пайплайн.
"""

from .config import (
    get_config, Config,
    PUNCT_LABELS, PARA_LABELS, CAP_LABELS, ACOUSTIC_FEATURES,
    PRETRAINED_PRESETS,
)
from .data import build_examples, parse_transcription, Example, load_fleurs
from .tokenizer import WordVocab
from .dataset import (
    BaselineDataset, PretrainedDataset, baseline_collate, pretrained_collate,
)
from .models import build_model, load_hf_tokenizer
from .train import train_model, set_seed
from .evaluate import evaluate, pretty_report
from .inference import PunctuationRestorer, STTPunctuationPipeline

__all__ = [
    "get_config", "Config",
    "PUNCT_LABELS", "PARA_LABELS", "CAP_LABELS", "ACOUSTIC_FEATURES",
    "PRETRAINED_PRESETS",
    "build_examples", "parse_transcription", "Example", "load_fleurs",
    "WordVocab",
    "BaselineDataset", "PretrainedDataset", "baseline_collate", "pretrained_collate",
    "build_model", "load_hf_tokenizer",
    "train_model", "set_seed",
    "evaluate", "pretty_report",
    "PunctuationRestorer", "STTPunctuationPipeline",
]
