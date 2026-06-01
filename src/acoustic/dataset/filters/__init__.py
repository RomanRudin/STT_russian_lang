from .duration_filter import apply as duration_filter
from .text_normalizer import apply as text_normalizer

# Registry for filters by name
FILTERS = {
    "duration_filter": duration_filter,
    "text_normalizer": text_normalizer,
}

def get_filter(name: str):
    """Return filter function by its registered name."""
    if name not in FILTERS:
        raise ValueError(f"Unknown filter: {name}. Available: {list(FILTERS.keys())}")
    return FILTERS[name]