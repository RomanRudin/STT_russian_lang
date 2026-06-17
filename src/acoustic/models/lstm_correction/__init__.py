from acoustic.models import MODEL_BUILDERS
from acoustic.models import GENERATE_METHODS


from .correction_model import build_lstm_correction
MODEL_BUILDERS.register("lstm_correction", build_lstm_correction)

from .correction_model import _lstm_generate

GENERATE_METHODS.register("lstm_correction", _lstm_generate)