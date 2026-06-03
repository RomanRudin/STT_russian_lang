from acoustic.utils.registry import PluginRegistry

MODEL_BUILDERS = PluginRegistry("model_builder")
GENERATE_METHODS = PluginRegistry("generate_method")


def get_model_builder(name: str):
    """Return a model builder callable by its registry key."""
    return MODEL_BUILDERS.get(name)

def get_generate_method(name: str):
    """Analogically"""
    return GENERATE_METHODS.get(name)