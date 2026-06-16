from acoustic.models import MODEL_BUILDERS
from acoustic.models import GENERATE_METHODS


from .correction_model import build_rut5
MODEL_BUILDERS.register("correction_model", build_rut5)

from .correction_model import _rut5_generate

GENERATE_METHODS.register("correction_model", _rut5_generate)