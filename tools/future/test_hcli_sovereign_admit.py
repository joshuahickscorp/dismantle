"""The sovereign loop's shape boundary: no reply may crash validate().

The loop has taken three crashes on the SHAPE of a reply rather than its content
(n_accepted missing on the parse-failure path; selected_work returned as a dict
and sliced as a list; fourteen iterations that parsed nothing). Each was fixed
where it was found. This is the test that would have caught all three at once:
every shape admit() can produce is fed to validate(), and validate must return a
uniform result for all of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hcli_sovereign as sov  # noqa: E402
from resident_output_contract import SOVEREIGN_REPLY_SCHEMA, admit  # noqa: E402

VALIDATE_KEYS = {"ok", "accepted", "rejected", "n_accepted", "n_rejected"}

# Every shape the body has actually produced, plus the ones it is free to.
REPLIES = [
    "",
    "   ",
    "not json at all",
    "```json\n{}\n```",
    "{}",
    "[]",
    "null",
    '{"selected_work": {"type": "PERTURB"}}',           # dict where a list goes
    '{"selected_work": "PERTURB"}',                     # string where a list goes
    '{"selected_work": [null, 3, "x"]}',                # junk members
    '{"selected_work": [{"type": "DELETE_EVERYTHING"}]}',
    '{"selected_work": [{"type": "PERTURB", "params": {}}]}',
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "up", "layer": "NaN", "fraction": "half"}}]}',
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "sideways", "layer": 3, "fraction": 0.5}}]}',
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "up", "layer": 999, "fraction": 0.5}}]}',
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "up", "layer": 0, "fraction": 99.0}}]}',
    '{"belief_update": "H1 dead", "live_hypotheses": [{"id": "H3"}]',  # truncated
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "up", "layer": 0, "fraction": 0.5}}]} then it kept talking '
    'and talking and talking and talking and talking and talking',
    '{"a": 1} {"b": 2}',
    '{"selected_work": [' + ','.join(
        ['{"type": "PERTURB", "params": {"tensor": "up", "layer": 1, '
         '"fraction": 0.5}}'] * 40) + ']',
]


@pytest.mark.parametrize("reply", REPLIES, ids=range(len(REPLIES)))
def test_no_reply_shape_can_crash_the_boundary(reply):
    adm = admit(reply, SOVEREIGN_REPLY_SCHEMA)
    obj = adm["value"] if adm["ok"] else adm["value"]
    v = sov.validate(obj)
    assert set(v) >= VALIDATE_KEYS, "validate must return one shape on every path"
    assert isinstance(v["accepted"], list)
    assert isinstance(v["rejected"], list)
    assert v["n_accepted"] == len(v["accepted"])
    assert v["n_rejected"] == len(v["rejected"])


@pytest.mark.parametrize("reply", REPLIES, ids=range(len(REPLIES)))
def test_admit_never_raises_and_holds_its_key_set(reply):
    adm = admit(reply, SOVEREIGN_REPLY_SCHEMA)
    assert set(adm) == set(admit("{}", SOVEREIGN_REPLY_SCHEMA))
    assert isinstance(adm["value"]["selected_work"], list), (
        "selected_work must be a list on every path - slicing a dict is the "
        "exact crash that killed iteration 2"
    )


def test_the_unparsed_path_still_carries_its_counts():
    """The first of the three crashes. Kept as a named regression."""
    v = sov.validate(None)
    assert v["ok"] is False
    assert v["n_accepted"] == 0 and v["n_rejected"] == 0


def test_a_dict_selected_work_is_coerced_not_sliced():
    """The third crash. A dict must become a one-item list, not a KeyError."""
    v = sov.validate({"selected_work": {
        "type": "PERTURB",
        "params": {"tensor": "up", "layer": 0, "fraction": 0.5}}})
    assert v["n_accepted"] == 1


def test_a_good_reply_is_still_accepted():
    """A boundary that rejects everything is not a boundary, it is a wall."""
    adm = admit(
        '{"belief_update": "H2 refuted", '
        '"live_hypotheses": [], "escalation_needed": false, '
        '"selected_work": [{"type": "PERTURB", "why": "down is most sensitive", '
        '"params": {"tensor": "down", "side": "rows", "layer": 12, '
        '"fraction": 0.4}}]}',
        SOVEREIGN_REPLY_SCHEMA)
    assert adm["ok"] is True, adm["missing"]
    v = sov.validate(adm["value"])
    assert v["n_accepted"] == 1
    assert v["accepted"][0]["params"]["tensor"] == "down"


def test_the_loop_imports_admit_rather_than_parsing_by_hand():
    src = Path(sov.__file__).read_text()
    assert "from resident_output_contract import admit" in src
    assert 'reask_kind = "narrow"' in src, (
        "the narrow re-ask is the point of wiring admit in; without it the "
        "loop still burns a whole turn re-running the scientific prompt"
    )


def test_the_narrow_reask_only_asks_for_what_is_missing():
    adm = admit('{"belief_update": "H1 dead", "live_hypotheses": [{"id": "H3"}]',
                SOVEREIGN_REPLY_SCHEMA)
    assert adm["parse"]["recovered"] is True
    assert adm["reask"]["needed"] is True
    frag = adm["reask"]["prompt_fragment"]
    assert "selected_work" in frag
    assert "belief_update" not in frag, (
        "re-asking for a field the body already supplied is the waste this "
        "path exists to avoid"
    )


def test_a_schema_incomplete_reply_that_carries_work_is_still_executed():
    """admit's schema requires `why` and `side`; validate does not need either
    to run the work. A reply missing only those must NOT cost a re-ask turn."""
    adm = admit(
        '{"belief_update": "x", "live_hypotheses": [], '
        '"escalation_needed": false, "selected_work": [{"type": "PERTURB", '
        '"params": {"tensor": "down", "layer": 12, "fraction": 0.4}}]}',
        SOVEREIGN_REPLY_SCHEMA)
    assert adm["ok"] is False
    assert set(adm["missing"]) <= {"selected_work.params.side", "selected_work.why"}
    assert sov.validate(adm["value"])["n_accepted"] == 1
    src = Path(sov.__file__).read_text()
    assert 'has_work = bool(adm["value"].get("selected_work"))' in src
