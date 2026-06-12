"""Shared utility helpers used across the Streamlit app."""


def _yaml_safe(obj):
    """Recursively convert tuples to lists for safe YAML round-trips.

    PyYAML serialises Python tuples as ``!!python/tuple`` tags, which
    ``yaml.safe_load`` refuses to deserialise.  This walks any nested
    dict/list/tuple structure and converts every tuple to a list so the
    output is compatible with ``yaml.safe_load``.

    Args:
        obj: Any Python object.  Dicts and sequences are walked
            recursively; all other values are returned unchanged.

    Returns:
        A new object with the same structure but all tuples replaced
        by lists.
    """
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    return obj
