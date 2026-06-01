"""
labels.py
=========
Единая схема меток для задачи восстановления пунктуации.

Модель решает три параллельные задачи (три "головы") на каждый токен:

1. PUNCT  — какой знак стоит ПОСЛЕ токена;
2. CASE   — регистр самого токена (капитализация);
3. PARA   — начинается ли с этого токена новый абзац (красная строка).

Двоеточие добавлено в набор PUNCT как COLON.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# 1. Пунктуация (знак ПОСЛЕ токена)
# --------------------------------------------------------------------------- #
PUNCT_LABELS = [
    "O",         # ничего
    "COMMA",     # ,
    "PERIOD",    # .
    "QUESTION",  # ?
    "EXCLAM",    # !
    "ELLIPSIS",  # …
    "COLON",     # :
]

# Текстовое представление каждого знака (что именно дописать к токену).
PUNCT_TO_CHAR = {
    "O": "",
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "EXCLAM": "!",
    "ELLIPSIS": "…",
    "COLON": ":",
}

# Какие символы во входном "чистом" тексте мапятся в какую метку пунктуации.
CHAR_TO_PUNCT = {
    ",": "COMMA",
    ".": "PERIOD",
    "?": "QUESTION",
    "!": "EXCLAM",
    "…": "ELLIPSIS",
    ":": "COLON",
}
# Многоточие из трёх точек тоже считаем ELLIPSIS — обрабатывается в препроцессинге.

# --------------------------------------------------------------------------- #
# 2. Капитализация (регистр токена)
# --------------------------------------------------------------------------- #
CASE_LABELS = [
    "LOWER",        # всё строчными:           привет
    "UPPER_FIRST",  # первая буква заглавная:   Привет
    "UPPER_ALL",    # всё прописными:           СССР, НАТО (аббревиатуры)
]

# --------------------------------------------------------------------------- #
# 3. Абзац (красная строка ПЕРЕД токеном)
# --------------------------------------------------------------------------- #
PARA_LABELS = [
    "NO_PARAGRAPH",  # продолжение текущего абзаца
    "PARAGRAPH",     # с этого токена начинается новый абзац
]

# --------------------------------------------------------------------------- #
# Индексы <-> метки
# --------------------------------------------------------------------------- #
PUNCT2ID = {l: i for i, l in enumerate(PUNCT_LABELS)}
ID2PUNCT = {i: l for l, i in PUNCT2ID.items()}

CASE2ID = {l: i for i, l in enumerate(CASE_LABELS)}
ID2CASE = {i: l for l, i in CASE2ID.items()}

PARA2ID = {l: i for i, l in enumerate(PARA_LABELS)}
ID2PARA = {i: l for l, i in PARA2ID.items()}

NUM_PUNCT = len(PUNCT_LABELS)
NUM_CASE = len(CASE_LABELS)
NUM_PARA = len(PARA_LABELS)

# Спец-индекс для подавления потерь на padding/subword-продолжениях.
IGNORE_INDEX = -100


def case_of_token(token: str) -> str:
    """Определяет метку регистра для исходного (правильного) токена."""
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return "LOWER"
    if all(c.isupper() for c in letters) and len(letters) > 1:
        return "UPPER_ALL"
    if letters[0].isupper():
        return "UPPER_FIRST"
    return "LOWER"


def apply_case(token: str, case_label: str) -> str:
    """Применяет предсказанный регистр к (строчному) токену."""
    if case_label == "UPPER_ALL":
        return token.upper()
    if case_label == "UPPER_FIRST":
        return token[:1].upper() + token[1:]
    return token


if __name__ == "__main__":
    print("PUNCT:", PUNCT_LABELS)
    print("CASE :", CASE_LABELS)
    print("PARA :", PARA_LABELS)
    for t in ["привет", "Москва", "СССР", "iPhone", "2024"]:
        print(f"  {t!r:12} -> {case_of_token(t)}")
