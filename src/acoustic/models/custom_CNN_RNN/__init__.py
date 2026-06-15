from .model import build_custom_model
from acoustic.models import MODEL_BUILDERS

MODEL_BUILDERS.register("custom_CNN_RNN", build_custom_model)

from .model import _custom_generate
from acoustic.models import GENERATE_METHODS

GENERATE_METHODS.register("custom_CNN_RNN", _custom_generate)