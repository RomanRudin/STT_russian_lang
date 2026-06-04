"""
config.py
=========
Единая точка конфигурации проекта восстановления пунктуации.

Здесь определены:
  * схемы меток для трёх голов (пунктуация / абзац / капитализация);
  * список и нормализация акустических признаков (late fusion, вариант A);
  * гиперпараметры моделей и обучения;
  * пути и имена предобученных моделей.

Все остальные модули импортируют отсюда, чтобы метки и размерности
гарантированно совпадали между data / dataset / models / inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# 1. Схемы меток (три независимые головы token-classification)
# ---------------------------------------------------------------------------

# Голова 1 — знак ПОСЛЕ токена.
PUNCT_LABELS: List[str] = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAM", "ELLIPSIS"]
# Символ, который реально дописывается после слова при восстановлении текста.
PUNCT_TO_CHAR: Dict[str, str] = {
    "O": "",
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "EXCLAM": "!",
    "ELLIPSIS": "…",
}
# Обратное отображение: символ -> метка (используется при разметке корпуса).
CHAR_TO_PUNCT: Dict[str, str] = {
    ",": "COMMA",
    ".": "PERIOD",
    "?": "QUESTION",
    "!": "EXCLAM",
    "…": "ELLIPSIS",
    "...": "ELLIPSIS",
}

# Голова 2 — абзац (красная строка) ПЕРЕД токеном.
PARA_LABELS: List[str] = ["NO_PARA", "PARA"]

# Голова 3 — капитализация токена.
CAP_LABELS: List[str] = ["LOWER", "CAP", "UPPER"]  # все строчные / Первая заглавная / ВСЕ заглавные


# Удобные индексы (label -> id и обратно) для каждой головы.
PUNCT2ID = {l: i for i, l in enumerate(PUNCT_LABELS)}
ID2PUNCT = {i: l for l, i in PUNCT2ID.items()}
PARA2ID = {l: i for i, l in enumerate(PARA_LABELS)}
ID2PARA = {i: l for l, i in PARA2ID.items()}
CAP2ID = {l: i for i, l in enumerate(CAP_LABELS)}
ID2CAP = {i: l for l, i in CAP2ID.items()}

NUM_PUNCT = len(PUNCT_LABELS)
NUM_PARA = len(PARA_LABELS)
NUM_CAP = len(CAP_LABELS)

# id, которым помечаются субтокены-продолжения (BERT) и паддинг — игнорируются в loss.
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# 2. Акустические признаки (late fusion, вариант A)
# ---------------------------------------------------------------------------
# Один вектор фиксированной длины на КАЖДОЕ слово. В инференсе те же признаки
# приходят из word-timestamps (forced alignment либо Whisper).
ACOUSTIC_FEATURES: List[str] = [
    "pause_before",   # длительность паузы перед словом, сек
    "pause_after",    # длительность паузы после слова, сек  <-- главный сигнал границ
    "word_duration",  # длительность самого слова, сек
    "speech_rate",    # темп: длительность / число символов (сек/символ)
    "f0_end_median",  # медиана F0 в конце слова, Гц (нормализованная) — для ?/!
    "f0_end_slope",   # наклон F0 в конце слова (рост -> вопрос)
    "energy_end",     # энергия (RMS) в конце слова
]
ACOUSTIC_DIM = len(ACOUSTIC_FEATURES)

# Грубая нормализация признаков к ~[0,1] / [-1,1]. Значения подобраны под русскую
# спонтанную/чтецкую речь; при наличии train-статистики лучше пересчитать на ней
# (см. data.compute_acoustic_stats).
ACOUSTIC_NORM: Dict[str, Dict[str, float]] = {
    "pause_before":  {"mean": 0.0,  "std": 0.30},
    "pause_after":   {"mean": 0.0,  "std": 0.30},
    "word_duration": {"mean": 0.35, "std": 0.20},
    "speech_rate":   {"mean": 0.06, "std": 0.03},
    "f0_end_median": {"mean": 160.0, "std": 60.0},
    "f0_end_slope":  {"mean": 0.0,  "std": 30.0},
    "energy_end":    {"mean": 0.05, "std": 0.05},
}


# ---------------------------------------------------------------------------
# 3. Гиперпараметры моделей
# ---------------------------------------------------------------------------

@dataclass
class LSTMConfig:
    vocab_size: int = 30000        # переопределяется реальным размером словаря
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    acoustic_dim: int = ACOUSTIC_DIM
    acoustic_hidden: int = 64      # размер MLP-энкодера акустики
    use_acoustic: bool = True


@dataclass
class TransformerConfig:
    vocab_size: int = 30000
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    max_len: int = 256
    acoustic_dim: int = ACOUSTIC_DIM
    acoustic_hidden: int = 64
    use_acoustic: bool = True


@dataclass
class PretrainedConfig:
    # Имена доступны на HuggingFace Hub.
    model_name: str = "DeepPavlov/rubert-base-cased"
    # Вторая лёгкая опция для слабого железа: "cointegrated/rubert-tiny2"
    acoustic_dim: int = ACOUSTIC_DIM
    acoustic_hidden: int = 64
    use_acoustic: bool = True
    dropout: float = 0.1


# Пресет «лёгкая предобученная» — выбирается в ноутбуке одним флагом.
PRETRAINED_PRESETS: Dict[str, str] = {
    "rubert-base": "DeepPavlov/rubert-base-cased",
    "rubert-tiny2": "cointegrated/rubert-tiny2",
}


# ---------------------------------------------------------------------------
# 4. Обучение
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 3e-4               # для предобученных понизить до ~2e-5 (см. train.py)
    weight_decay: float = 0.01
    epochs: int = 5
    max_len: int = 256             # максимум слов/субтокенов в примере
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1
    seed: int = 42
    device: str = "cuda"           # автоматически падает на cpu, если cuda недоступна
    num_workers: int = 0
    # Раздельные lr для предобученных моделей (используются при is_pretrained=True).
    encoder_lr: float = 3e-5       # мягкое дообучение энкодера RuBERT
    head_lr: float = 3e-3          # быстрее учим новые головы + акустический энкодер
    # Веса трёх лоссов (пунктуация важнее всего).
    w_punct: float = 1.0
    w_para: float = 0.5
    w_cap: float = 0.5

    # --- борьба с дисбалансом классов (главная проблема на этой задаче) ---
    # Тип лосса: "ce" (CrossEntropy с весами классов) или "focal" (Focal Loss).
    # Focal Loss обычно лучше при сильном перекосе в сторону класса O.
    loss_type: str = "focal"
    focal_gamma: float = 2.0       # сила фокусировки на трудных примерах

    # Веса классов пунктуации. Если auto_class_weights=True, веса считаются
    # автоматически по обратной частоте классов в train (см. train.compute_class_weights)
    # и эти ручные значения игнорируются.
    auto_class_weights: bool = True
    punct_class_weights: List[float] = field(
        default_factory=lambda: [0.3, 1.0, 1.0, 2.0, 2.0, 2.0]
    )
    # То же для голов para/cap (по умолчанию автоподбор при auto_class_weights).
    balance_para_cap: bool = True


# ---------------------------------------------------------------------------
# 5. Данные / пути
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # --- основной корпус: M-AILABS (русские аудиокниги) ---
    # Источник — LibriVox/Gutenberg: художественные тексты с ПОЛНОЙ пунктуацией
    # (запятые, точки, ? ! …) и реальными абзацами. Поэтому, в отличие от FLEURS,
    # классы QUESTION/EXCLAM/ELLIPSIS и голова PARA получают реальный сигнал.
    dataset_name: str = "google/fleurs"   # оставлен для совместимости/смешивания
    lang: str = "ru_ru"

    # Имена загрузок M-AILABS (ru) на HuggingFace Hub. Перебираются по порядку:
    # берётся первая, которая успешно загрузится. Если ни одна не сработает,
    # load_mailabs дополнительно ищет датасет по Hub API автоматически.
    # Схема `psiyou/m-ailabs-XX_XX` подтверждена для других языков (it_IT, и т.п.).
    mailabs_repos: List[str] = field(default_factory=lambda: [
        "psiyou/m-ailabs-ru_RU",
        "gigant/m-ailabs_speech_dataset_ru",
        "Vikhrmodels/m-ailabs_ru",
    ])
    # Разрешить автопоиск датасета по Hub API, если список выше не сработал.
    mailabs_autosearch: bool = True
    # Возможные имена текстового поля в разных загрузках M-AILABS.
    text_field_candidates: List[str] = field(default_factory=lambda: [
        "sentence", "transcription", "raw_transcription", "text",
    ])
    # M-AILABS обычно идёт одним split 'train' — делим сами.
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    cache_dir: str = "./.cache"
    # Каталог, куда forced-aligner кладёт пословные тайминги (json на сэмпл).
    alignment_dir: str = "./.alignments"
    # Если выравнивание недоступно — обучаемся в text-only режиме (фичи = нули).
    allow_text_only: bool = True
    sample_rate: int = 16000
    # Минимальная длина примера в словах (короткие реплики отбрасываем).
    min_words: int = 3


# Единый удобный контейнер.
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    pretrained: PretrainedConfig = field(default_factory=PretrainedConfig)


def get_config() -> Config:
    return Config()
