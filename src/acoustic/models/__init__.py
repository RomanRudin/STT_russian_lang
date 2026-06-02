from acoustic.utils.registry import PluginRegistry

MODEL_BUILDERS = PluginRegistry("model_builder")

def get_model_builder(name: str):
    """Return a model builder callable by its registry key."""
    return MODEL_BUILDERS.get(name)