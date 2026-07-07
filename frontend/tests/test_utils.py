"""
tests/test_utils.py

Tests for sentinel_frontend.utils — _yaml_safe tuple-to-list conversion.
"""

import yaml
import pytest

from sentinel_frontend.utils import _yaml_safe


# ---------------------------------------------------------------------------
# _yaml_safe
# ---------------------------------------------------------------------------

def test_yaml_safe_top_level_tuple_becomes_list():
    """A bare tuple at the top level is converted to a list."""
    result = _yaml_safe((1, 2, 3))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_yaml_safe_nested_tuple_in_dict_value():
    """A tuple nested inside a dict value is recursively converted."""
    result = _yaml_safe({"key": (10, 20)})
    assert result == {"key": [10, 20]}
    assert isinstance(result["key"], list)


def test_yaml_safe_nested_tuple_in_list():
    """A tuple nested inside a list is recursively converted."""
    result = _yaml_safe([(1, 2), (3, 4)])
    assert result == [[1, 2], [3, 4]]
    assert all(isinstance(v, list) for v in result)


def test_yaml_safe_existing_lists_pass_through():
    """Lists (and lists nested in dicts) are returned unchanged in type."""
    obj = {"a": [1, 2, 3], "b": [[4, 5]]}
    result = _yaml_safe(obj)
    assert result == obj
    assert isinstance(result["a"], list)
    assert isinstance(result["b"][0], list)


@pytest.mark.parametrize(
    "scalar",
    [42, "hello", 3.14, None],
    ids=["int", "str", "float", "none"],
)
def test_yaml_safe_scalars_pass_through(scalar):
    """Scalar values (int, str, float, None) are returned unmodified."""
    assert _yaml_safe(scalar) is scalar or _yaml_safe(scalar) == scalar


def test_yaml_safe_round_trip_via_yaml():
    """After _yaml_safe, yaml.dump output round-trips through yaml.safe_load without error."""
    obj = {
        "bounds": (0.0, 0.0, 1.0, 1.0),
        "tags": ["sentinel", "rgb"],
        "nested": {"offsets": (10, 20)},
        "count": 3,
    }
    safe_obj = _yaml_safe(obj)
    dumped = yaml.dump(safe_obj)
    reloaded = yaml.safe_load(dumped)
    assert reloaded["bounds"] == [0.0, 0.0, 1.0, 1.0]
    assert reloaded["nested"]["offsets"] == [10, 20]
    assert reloaded["count"] == 3
