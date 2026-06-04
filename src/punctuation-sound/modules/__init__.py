"""
modules — пакет восстановления пунктуации (мультимодальный, late fusion / вариант A).

Публичный API для ноутбука и встраивания в SpeechToText-пайплайн.
"""

# Версия пакета. Если в ноутбуке modules.__version__ < "2.3", значит загружена
# СТАРАЯ версия модулей (нужно обновить файлы в папке modules/ и перезапустить ядро).
__version__ = "2.3-mailabs"

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
      4) доступность библиотеки datasets и имена M-AILABS.
    """
    import sys
    print(f"modules.__version__ = {__version__}")
    print(f"python (ядро)       = {sys.executable}")

    # 1) версия config — есть ли новые имена датасета
    from .config import DataConfig
    repos = DataConfig().mailabs_repos
    print(f"mailabs_repos       = {repos}")
    if any("mailabs/ru_RU" == r for r in repos) or "psiyou/m-ailabs-ru_RU" not in repos:
        print("  !! ВНИМАНИЕ: загружена СТАРАЯ версия modules. Обновите файлы в папке")
        print("     modules/ последней версией и перезапустите ядро (Kernel -> Restart).")
    else:
        print("  ок: версия модулей актуальная.")

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

    # 4) Hub доступность (быстрая проверка)
    try:
        from huggingface_hub import HfApi
        names = [d.id for d in HfApi().list_datasets(search="m-ailabs", limit=10)]
        print(f"  M-AILABS на Hub (поиск): {names[:10] if names else 'ничего не найдено'}")
    except Exception as e:
        print(f"  поиск по Hub недоступен ({type(e).__name__}: {e}).")
        print("  Возможен ограниченный доступ к huggingface.co (прокси/файрвол).")
