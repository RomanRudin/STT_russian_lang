from .model import build_wav2vec2
from acoustic.models import MODEL_BUILDERS

MODEL_BUILDERS.register("wav2vec2", build_wav2vec2)

from .model import _wav2vec2_generate
from acoustic.models import GENERATE_METHODS

GENERATE_METHODS.register("wav2vec2", _wav2vec2_generate)