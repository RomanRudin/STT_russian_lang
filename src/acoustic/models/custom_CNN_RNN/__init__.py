from acoustic.models import MODEL_BUILDERS
from acoustic.models import GENERATE_METHODS


from .model import build_custom_model
MODEL_BUILDERS.register("custom_CNN_RNN", build_custom_model)

from .model import _custom_generate

GENERATE_METHODS.register("custom_CNN_RNN", _custom_generate)