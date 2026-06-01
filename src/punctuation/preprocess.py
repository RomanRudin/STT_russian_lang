"""
preprocess.py
=============
Преобразование между "сырым" текстом (с пунктуацией, регистром, абзацами)
и обучающим представлением: список токенов + три параллельных списка меток.

Ключевые функции:
  text_to_labeled(text)  -> (tokens, punct_labels, case_labels, para_labels)
  labels_to_text(...)    -> восстановленный текст с пунктуацией

Токены — это слова. На вход модели подаётся "обеднённый" текст:
все слова в нижнем регистре, без пунктуации, в одну строку.
Модель должна восстановить то, что выбросил degrade_text().
"""

from __future__ import annotations

import re
from typing import List, Tuple

from utils.labels import (
    CHAR_TO_PUNCT,
    PUNCT_TO_CHAR,
    case_of_token,
    apply_case,
)

# Знаки, которые мы восстанавливаем (нужны для парсинга исходного текста).
_PUNCT_CHARS = set(CHAR_TO_PUNCT.keys()) | {"…"}
# Регулярка: слово (буквы/цифры/дефис) либо одиночный знак препинания.
_TOKEN_RE = re.compile(r"\w[\w\-]*|[,.!?:…]+", re.UNICODE)


def normalize_ellipsis(text: str) -> str:
    """Три и более точек -> символ многоточия …"""
    return re.sub(r"\.{2,}", "…", text)


def split_paragraphs(text: str) -> List[str]:
    """Разбивает текст на абзацы по пустым строкам / переносам."""
    parts = re.split(r"\n\s*\n|\n", text)
    return [p.strip() for p in parts if p.strip()]


def text_to_labeled(
    text: str,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Размеченный текст -> (tokens, punct, case, para).

    tokens — слова в НИЖНЕМ регистре без пунктуации (вход модели);
    punct  — знак ПОСЛЕ каждого слова;
    case   — регистр исходного слова;
    para   — 'PARAGRAPH' для первого слова каждого абзаца (кроме самого первого
             слова текста, которое всегда NO_PARAGRAPH — это просто начало).
    """
    tokens: List[str] = []
    punct: List[str] = []
    case: List[str] = []
    para: List[str] = []

    paragraphs = split_paragraphs(text)
    for p_idx, paragraph in enumerate(paragraphs):
        paragraph = normalize_ellipsis(paragraph)
        raw = _TOKEN_RE.findall(paragraph)

        first_word_in_para = True
        for tok in raw:
            if tok[0] in _PUNCT_CHARS:
                # знак препинания: приписываем его предыдущему слову
                if not tokens:
                    continue  # текст не может начинаться со знака
                # многосимвольное "?!" -> берём первый значимый знак
                ch = tok[0] if tok[0] != "." or len(tok) == 1 else "…"
                if tok == "…":
                    ch = "…"
                label = CHAR_TO_PUNCT.get(ch, "O")
                # перезаписываем только если текущая метка пустая
                if punct[-1] == "O":
                    punct[-1] = label
            else:
                tokens.append(tok.lower())
                punct.append("O")
                case.append(case_of_token(tok))
                # первое слово абзаца (но не самого первого абзаца) -> PARAGRAPH
                if first_word_in_para and p_idx > 0:
                    para.append("PARAGRAPH")
                else:
                    para.append("NO_PARAGRAPH")
                first_word_in_para = False

    return tokens, punct, case, para


def degrade_text(text: str) -> str:
    """Готовит ВХОД для инференса: нижний регистр, без пунктуации, одна строка."""
    tokens, _, _, _ = text_to_labeled(text)
    return " ".join(tokens)


def labels_to_text(
    tokens: List[str],
    punct: List[str],
    case: List[str],
    para: List[str],
) -> str:
    """Собирает финальный текст с пунктуацией, регистром и абзацами."""
    out_parts: List[str] = []
    for i, tok in enumerate(tokens):
        word = apply_case(tok, case[i])
        # новый абзац -> перенос строки перед словом
        if para[i] == "PARAGRAPH" and out_parts:
            out_parts.append("\n")
        elif out_parts and not out_parts[-1].endswith("\n"):
            out_parts.append(" ")
        out_parts.append(word + PUNCT_TO_CHAR.get(punct[i], ""))

    text = "".join(out_parts)
    # Косметика: заглавная буква после .?!… и в начале текста.
    text = _autocapitalize_after_sentence(text)
    return text.strip()


def _autocapitalize_after_sentence(text: str) -> str:
    """
    Подстраховка регистра: первая буква текста и первая буква после
    конечного знака (. ? ! …) делается заглавной, если модель регистр не угадала.
    """
    def cap_first_alpha(s: str) -> str:
        for i, c in enumerate(s):
            if c.isalpha():
                return s[:i] + c.upper() + s[i + 1 :]
        return s

    # начало текста
    text = cap_first_alpha(text)
    # после конечных знаков
    def repl(m: re.Match) -> str:
        return m.group(1) + cap_first_alpha(m.group(2))

    text = re.sub(r"([.?!…]\s+)(\S+)", repl, text)
    return text


if __name__ == "__main__":
    sample = (
        "привет как дела? я давно тебя не видел.\n"
        "вот что я подумал: надо встретиться. ты согласен?"
    )
    toks, pu, ca, pa = text_to_labeled(sample)
    print("tokens:", toks)
    print("punct :", pu)
    print("case  :", ca)
    print("para  :", pa)
    print("\ndegrade:", degrade_text(sample))
    print("\nrebuilt:")
    print(labels_to_text(toks, pu, ca, pa))
