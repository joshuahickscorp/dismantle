"""A schema rejection that names only what is ABSENT teaches nothing to a backend
that got the shape right and drifted one key.

Measured against the sealed 27B resident on 2026-08-26: it wrote "consequential"
on obligations[0] and "consequent" on [1], [2] and [3]. The rejection said
"missing required property 'consequential'" six times and the model never
converged, because nothing in the message pointed at the key it had written.
"""
import pytest

from hcli.backends import _edit_distance, _near_miss_key, validate_against_schema

SCHEMA = {"type": "object",
          "properties": {"consequential": {"type": "boolean"},
                         "id": {"type": "string"},
                         "angles": {"type": "array"}},
          "required": ["consequential", "id"]}


def test_ANTI_VACUITY_a_valid_object_is_not_rejected():
    assert validate_against_schema(
        {"consequential": True, "id": "x"}, SCHEMA) is None


def test_THE_MEASURED_DRIFT_IS_NAMED():
    err = validate_against_schema({"id": "x", "consequent": True}, SCHEMA)
    assert "consequent" in err and "consequential" in err


def test_A_PLAIN_OMISSION_IS_NOT_DECORATED_WITH_A_GUESS():
    """Nothing resembling the key is present, so the message must not invent one."""
    err = validate_against_schema({"id": "x"}, SCHEMA)
    assert "you wrote" not in err


def test_AN_UNRELATED_EXTRA_KEY_IS_NOT_CALLED_A_MISSPELLING():
    err = validate_against_schema(
        {"id": "x", "completely_unrelated_field": 1}, SCHEMA)
    assert "you wrote" not in err, err


def test_A_LEGITIMATE_SIBLING_PROPERTY_IS_NEVER_THE_NEAR_MISS():
    """`angles` is a real property of this schema. A required key going missing
    does not make a sibling that IS in the schema a misspelling of it."""
    assert _near_miss_key("consequential", {"angles": []}, SCHEMA) is None


def test_SHORT_KEYS_CANNOT_COLLIDE_WITH_EACH_OTHER():
    """The threshold scales with name length, so `id` does not match `is`."""
    sch = {"type": "object", "properties": {"id": {}}, "required": ["id"]}
    assert _near_miss_key("id", {"ix": 1}, sch) is None


@pytest.mark.parametrize("a,b,d", [("", "", 0), ("a", "", 1), ("abc", "abc", 0),
                                   ("consequential", "consequent", 3),
                                   ("kitten", "sitting", 3)])
def test_the_distance_is_the_real_one(a, b, d):
    assert _edit_distance(a, b) == d
