from .model import build_hubert
from acoustic.models import MODEL_BUILDERS

MODEL_BUILDERS.register("hubert", build_hubert)

from .model import _hubert_generate
from acoustic.models import GENERATE_METHODS

GENERATE_METHODS.register("hubert", _hubert_generate)