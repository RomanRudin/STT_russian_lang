import importlib
import logging
from typing import Any, Tuple, Optional

from acoustic.models import MODEL_BUILDERS, get_model_builder
from acoustic.models.collators import get_collator_class
from acoustic.augmentations import build_augmentations

logger = logging.getLogger(__name__)


def build_model(cfg: dict) -> Tuple[Any, Any, Optional[Any]]:
    """
    Build model, processor, and optional data collator.

    cfg['model'] must contain either:
        - 'builder': key in MODEL_BUILDERS (e.g. "whisper"), or
        - 'import_path': dotted path like "whisper.model.build_whisper"
          (first part before dot is used as builder key).

    If cfg['model'] contains 'collator', that name is looked up in
    the collator registry and an instance is created with `processor`.
    Additional collator parameters can be passed via
    cfg['model'].get('collator_params', {}).
    """
    # Determine builder key
    builder_key = cfg['model'].get('builder')
    if builder_key is None:
        # Fallback to import_path
        import_path = cfg['model'].get('import_path', '')
        if '.' in import_path:
            builder_key = import_path.split('.')[0]
        else:
            builder_key = import_path

    if not builder_key:
        raise ValueError("No model builder specified in config (set 'builder' or 'import_path')")

    # Ensure the module is registered (auto-import if needed)
    if builder_key not in MODEL_BUILDERS:
        logger.info("Builder '%s' not registered, trying to import package acoustic.models.%s", builder_key, builder_key)
        try:
            importlib.import_module(f"acoustic.models.{builder_key}")
        except ImportError:
            raise ImportError(
                f"Could not import module acoustic.models.{builder_key}. "
                f"Make sure the package exists and registers itself into MODEL_BUILDERS."
            )
        if builder_key not in MODEL_BUILDERS:
            raise ValueError(
                f"Module acoustic.models.{builder_key} imported but did not register "
                f"builder '{builder_key}'"
            )

    # Build model and processor
    builder = get_model_builder(builder_key)
    model, processor = builder(cfg)

    # Load data collator if configured
    data_collator = None
    collator_name = cfg['model'].get('collator')
    if collator_name:
        collator_cls = get_collator_class(collator_name)
        collator_kwargs = cfg['model'].get('collator_params', {}).copy()
        # Extract and build augmentations
        aug_config = collator_kwargs.pop('augmentations', {})
        augmentations = build_augmentations(aug_config) if aug_config else []
        data_collator = collator_cls(processor, augmentations=augmentations, **collator_kwargs)
        logger.info("Using collator '%s' with augmentations: %s",
                    collator_name,
                    list(aug_config.keys()) if aug_config else "none")
    else:
        logger.info("No collator specified; using default DataCollatorForSeq2Seq in trainer")

    return model, processor, data_collator