"""
modules — пакет восстановления пунктуации (мультимодальный, late fusion / вариант A).

Публичный API для ноутбука и встраивания в SpeechToText-пайплайн.
"""

# Версия пакета. Если в ноутбуке modules.__version__ < "3.2", значит загружена
# СТАРАЯ версия модулей (нужно обновить файлы в папке modules/ и перезапустить ядро).
__version__ = "3.2-ruls"

from .config import (
    get_config, Config,
    PUNCT_LABELS, PARA_LABELS, CAP_LABELS, ACOUSTIC_FEATURES,
    PRETRAINED_PRESETS,
)
from .data import build_examples, parse_transcription, Example, load_fleurs, examples_from_pkl
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
    "build_examples", "parse_transcription", "Example", "load_fleurs", "examples_from_pkl",
    "WordVocab",
    "BaselineDataset", "PretrainedDataset", "baseline_collate", "pretrained_collate",
    "build_model", "load_hf_tokenizer",
    "train_model", "set_seed",
    "evaluate", "pretty_report",
    "PunctuationRestorer", "STTPunctuationPipeline",
    "diagnose", "__version__",
]


def diagnose() -> None:
    """
    Самодиагностика окружения. Запустите ПЕРЕД основным прогоном:
        import modules; modules.diagnose()

    Проверяет:
      1) версию загруженных модулей (та ли она, и не кэш ли старая);
      2) из какого python запущено ядро (для отладки «установил, но не видит»);
      3) импорт forced-aligner + объясняет, почему не вышло;
      4) доступность библиотеки datasets и источника RuLS.
    """
    import sys
    print(f"modules.__version__ = {__version__}")
    print(f"python (ядро)       = {sys.executable}")

    # 1) версия config — RuLS-источник и приоритет поля с пунктуацией
    from .config import DataConfig
    dc = DataConfig()
    has_ruls = hasattr(dc, "ruls_archive_urls")
    print(f"ruls_repo           = {getattr(dc, 'ruls_repo', '—')}")
    print(f"text_field[0]       = {dc.text_field_candidates[0]}")
    if not has_ruls or dc.text_field_candidates[0] != "text_no_preprocessing":
        print("  !! ВНИМАНИЕ: загружена СТАРАЯ версия modules (нет RuLS или не то текстовое поле).")
        print("     Обновите файлы в папке modules/ и перезапустите ядро (Kernel -> Restart).")
    else:
        print("  ок: версия модулей актуальная (RuLS, поле text_no_preprocessing).")

    # 2) forced-aligner
    print("\nforced-aligner:")
    try:
        import ctc_forced_aligner  # noqa: F401
        loc = getattr(ctc_forced_aligner, "__file__", "?")
        print(f"  ок: ctc_forced_aligner импортируется ({loc})")
    except ModuleNotFoundError:
        print("  !! НЕ установлен В ЭТОМ окружении (ModuleNotFoundError).")
        print("     Установите тем же python, что у ядра (см. путь выше), прямо в ячейке:")
        print("     !{sys.executable} -m pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git")
        print("     затем перезапустите ядро.")
    except Exception as e:
        print(f"  !! импорт падает с ошибкой (не ModuleNotFound): {type(e).__name__}: {e}")
        print("     Часто это отсутствие ffmpeg или несобранная зависимость. Пришлите этот текст.")

    # 3) datasets
    print("\nдатасеты:")
    try:
        import datasets  # noqa: F401
        print(f"  ок: datasets {datasets.__version__}")
    except Exception as e:
        print(f"  !! библиотека datasets недоступна: {e}")
        print("     pip install 'datasets>=2.19,<2.21'")

    # 4) Источник RuLS
    print("\nRuLS:")
    print("  Основной путь — официальный архив OpenSLR (зеркала US/EU/CN), не HF Hub.")
    print("  Рекомендуется: python prepare_ruls.py  (см. README), затем examples_from_pkl(...).")
    try:
        from huggingface_hub import HfApi  # noqa: F401
        print("  huggingface_hub доступен (можно пробовать и HF-путь).")
    except Exception:
        print("  huggingface_hub недоступен — используйте архивный путь (prepare_ruls.py).")
