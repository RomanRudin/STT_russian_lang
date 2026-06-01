import yaml
import ast
from typing import Dict, Any, Optional, List
import os
from copy import deepcopy

def deep_merge(base: Dict, updates: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    result = deepcopy(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML config and apply dot‑notation overrides.

    Args:
        config_path: Path to the base YAML config.
        overrides: Dictionary with keys like "training.learning_rate" and values.

    Returns:
        Merged configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            parts = key.split('.')
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

    return config

def apply_cli_overrides(cfg: Dict[str, Any], overrides_list: List[str]) -> Dict[str, Any]:
    """
    Apply --set KEY=VALUE overrides from command line.

    Args:
        cfg: Configuration dictionary.
        overrides_list: List of strings like "training.learning_rate=5e-6".

    Returns:
        Modified configuration dictionary.
    """
    for item in overrides_list:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, _, raw = item.partition("=")
        keys = key.strip().split(".")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return cfg

def save_config_snapshot(cfg: Dict[str, Any], output_dir: str) -> None:
    """Save a copy of the configuration to output_dir for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)