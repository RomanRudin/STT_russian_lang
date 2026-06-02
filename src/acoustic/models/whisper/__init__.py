from .model import build_whisper
from acoustic.models import MODEL_BUILDERS

MODEL_BUILDERS.register("whisper", build_whisper)