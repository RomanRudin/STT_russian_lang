"""
src/pipeline.py

Inference pipeline: audio file → transcribed text.

Usage (Python):
    pipeline = RussianSTTPipeline("./checkpoints/whisper-ru")
    text = pipeline.transcribe("audio.wav")

Usage (CLI):
    python -m src.pipeline --model ./checkpoints/whisper-ru --audio speech.wav
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voice Activity Detection (simple energy-based chunker)
# ---------------------------------------------------------------------------

class SimpleVAD:
    """
    Splits long audio into ≤30 s chunks at silence boundaries.
    Whisper's hard limit is 30 s; this lets us handle arbitrary-length files.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_sec: float = 28.0,       # target chunk length
        min_silence_ms: float = 300.0, # minimum silence gap for a split
        silence_threshold: float = 0.01,
    ):
        self.sr = sample_rate
        self.chunk_len = int(chunk_sec * sample_rate)
        self.min_silence = int(min_silence_ms / 1000 * sample_rate)
        self.threshold = silence_threshold

    def split(self, waveform: np.ndarray) -> List[np.ndarray]:
        """
        Args:
            waveform: 1-D float32 numpy array at self.sr

        Returns:
            List of 1-D numpy chunks, each ≤ chunk_len samples
        """
        if len(waveform) <= self.chunk_len:
            return [waveform]

        chunks = []
        start = 0

        while start < len(waveform):
            end = start + self.chunk_len

            if end >= len(waveform):
                chunks.append(waveform[start:])
                break

            # Look backward from `end` for a silence region
            window = waveform[end - self.min_silence: end]
            energy = np.abs(window)

            if (energy < self.threshold).any():
                # Find rightmost silence sample in the window
                silence_idx = (energy < self.threshold).nonzero()[0][-1]
                end = end - self.min_silence + silence_idx

            chunks.append(waveform[start:end])
            start = end

        return chunks


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class RussianSTTPipeline:
    """
    Wraps the fine-tuned Whisper model for easy inference.

    Example:
        pipeline = RussianSTTPipeline("./checkpoints/whisper-ru")
        print(pipeline.transcribe("interview.wav"))
    """

    TARGET_SR = 16_000

    def __init__(
        self,
        model_dir: str = "./checkpoints/whisper-ru",
        device: Optional[str] = None,
        language: str = "russian",
        task: str = "transcribe",
        beam_size: int = 5,
        use_vad: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model from %s on %s …", model_dir, self.device)

        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(self.device).eval()

        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )
        self.beam_size = beam_size
        self.vad = SimpleVAD() if use_vad else None

    # ------------------------------------------------------------------

    @torch.inference_mode()
    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """
        Transcribe an audio file and return the full text.

        Args:
            audio_path: Path to any audio format supported by torchaudio.

        Returns:
            Transcribed string (with punctuation if the model produces it).
        """
        t0 = time.perf_counter()
        waveform = self._load_audio(audio_path)

        chunks = self.vad.split(waveform) if self.vad else [waveform]
        logger.info("Audio split into %d chunk(s)", len(chunks))

        transcripts = []
        for i, chunk in enumerate(chunks):
            text = self._transcribe_chunk(chunk)
            logger.debug("Chunk %d: %s", i, text)
            transcripts.append(text)

        result = " ".join(t.strip() for t in transcripts if t.strip())
        elapsed = time.perf_counter() - t0
        logger.info("Done in %.2f s | RTF: %.2fx", elapsed, elapsed / (len(waveform) / self.TARGET_SR))
        return result

    # ------------------------------------------------------------------

    def _load_audio(self, path: Union[str, Path]) -> np.ndarray:
        """Load, resample to 16 kHz, convert to mono float32 numpy array."""
        wav, sr = torchaudio.load(str(path))

        if sr != self.TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, self.TARGET_SR)

        if wav.shape[0] > 1:          # stereo → mono
            wav = wav.mean(dim=0, keepdim=True)

        return wav.squeeze(0).numpy().astype(np.float32)

    def _transcribe_chunk(self, chunk: np.ndarray) -> str:
        """Run Whisper on a single ≤30 s chunk."""
        inputs = self.processor(
            chunk,
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)

        predicted_ids = self.model.generate(
            input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            num_beams=self.beam_size,
            no_repeat_ngram_size=3,
        )

        return self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Transcribe Russian audio with fine-tuned Whisper")
    parser.add_argument("--model",  required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--audio",  required=True, help="Path to audio file")
    parser.add_argument("--device", default=None,  help="'cuda' or 'cpu' (auto-detected if omitted)")
    parser.add_argument("--beam",   type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--no_vad", action="store_true", help="Disable VAD chunking (for short clips)")
    args = parser.parse_args()

    pipeline = RussianSTTPipeline(
        model_dir=args.model,
        device=args.device,
        beam_size=args.beam,
        use_vad=not args.no_vad,
    )
    transcript = pipeline.transcribe(args.audio)
    print("\nTranscript:")
    print("─" * 60)
    print(transcript)
    print("─" * 60)


if __name__ == "__main__":
    main()
