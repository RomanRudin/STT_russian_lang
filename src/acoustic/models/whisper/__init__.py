from .model import build_whisper
from acoustic.models import MODEL_BUILDERS

MODEL_BUILDERS.register("whisper", build_whisper)

from .model import _whisper_generate
from acoustic.models import GENERATE_METHODS

GENERATE_METHODS.register("whisper", _whisper_generate)