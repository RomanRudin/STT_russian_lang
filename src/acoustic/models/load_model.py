import importlib
from typing import Any, Tuple, Optional

def build_model(cfg: dict) -> Tuple[Any, Any, Optional[Any]]:
    """
    Dynamically load and build a model, processor, and optional data collator.

    Args:
        cfg: Configuration dictionary. Must contain 'model.import_path'
             (e.g., "whisper.model.build_whisper").

    Returns:
        Tuple of (model, processor, data_collator). data_collator may be None.
    """
    import_path = cfg['model']['import_path']
    module_path, func_name = import_path.rsplit('.', 1)
    full_module = f"acoustic.models.{module_path}"
    module = importlib.import_module(full_module)
    builder = getattr(module, func_name)
    model, processor = builder(cfg)

    # Try to get a data collator from the module (optional)
    data_collator = None
    if hasattr(module, 'get_data_collator'):
        data_collator = module.get_data_collator(processor)

    return model, processor, data_collator