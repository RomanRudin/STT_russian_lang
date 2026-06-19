from .model import build_wavlm, build_wavlm
from acoustic.models import MODEL_BUILDERS

MODEL_BUILDERS.register("wavlm", build_wavlm)

from .model import _wavlm_generate
from acoustic.models import GENERATE_METHODS

GENERATE_METHODS.register("wavlm", _wavlm_generate)