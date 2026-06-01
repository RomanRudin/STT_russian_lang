"""
stt_pipeline.py
===============
Адаптер для встраивания модели пунктуации в SpeechToText-пайплайн как
ВТОРОГО звена. Модель чисто текстовая, паузы из звука не используются.

    звук -> [Whisper] -> текст без пунктуации -> [PunctuationRestorer] -> текст

STTPunctuationPipeline принимает выход STT в виде:
  * строки                          -> используется как есть;
  * dict {"text": "..."}            -> берётся поле text;
  * dict {"words": [...]}           -> слова склеиваются в строку.
"""

from __future__ import annotations

from typing import Union, Dict, List

from utils.inference import PunctuationRestorer


class STTPunctuationPipeline:
    def __init__(self, restorer: PunctuationRestorer):
        self.restorer = restorer

    def __call__(self, stt_output: Union[str, Dict]) -> str:
        text = self._extract_text(stt_output)
        return self.restorer(text)

    @staticmethod
    def _extract_text(stt_output: Union[str, Dict]) -> str:
        if isinstance(stt_output, str):
            return stt_output
        if isinstance(stt_output, dict):
            if "text" in stt_output and stt_output["text"]:
                return stt_output["text"]
            if "words" in stt_output:
                words: List[str] = stt_output["words"]
                return " ".join(words)
        raise ValueError("Неподдерживаемый формат выхода STT")


# Пример подключения реального Whisper (псевдокод, не выполняется офлайн):
#
#   import whisper
#   asr = whisper.load_model("large-v3")
#   res = asr.transcribe("audio.wav", language="ru")
#   pipeline = STTPunctuationPipeline(restorer)
#   final_text = pipeline({"text": res["text"]})
