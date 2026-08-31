"""BYTE_COUNT x ORGAN_AVERAGE_RATE is not a valid cost model.

The 71 ladder credited quantize_aux_u8 with 1.99 TPS and group_size_256 with
3.08 by multiplying a byte count by the MLP organ average. ECONOMICS_CALIBRATION
dropped fractions of each stream and timed it: codes_keep_50 is faster than
2*MAD, aux_keep_50 is not. Both levers are 0.00.

The scar is EMITTED BY THE PRODUCER, not written into the receipt, because a
fix that lives only in a generated artifact has a lifetime of one rebuild - the
MLP_FUNCTION_REPLACEMENT umbrella scar was deleted exactly that way today.
"""
from __future__ import annotations

import json

from tools.future import _common as common
from tools.future import executable_economics as ee
from tools.future import negative_index as ni

REL = "receipts/future/ECONOMICS_CALIBRATION.json"
FAMILY = "BYTE_COUNT_TIMES_ORGAN_AVERAGE"


def _doc():
    return json.loads((common.REPO / REL).read_text())


def test_the_producer_emits_the_scar_not_the_receipt():
    raw = json.loads((common.REPO / "receipts/future/_ECONOMICS_CALIBRATION_raw.json").read_text())
    built = ee.calibrate_from_raw(raw)
    assert [s["family"] for s in built["scars"]] == [FAMILY], (
        "a rebuild must recreate this scar; if it only lives in the checked-in "
        "receipt it dies the next time anyone regenerates"
    )


def test_the_landed_receipt_carries_it_too():
    assert [s["family"] for s in _doc()["scars"]] == [FAMILY]


def test_the_negative_index_can_actually_see_it():
    """A scar the index cannot see prunes nothing."""
    assert REL in ni.LANDED_SCIENCE_RELS if hasattr(ni, "LANDED_SCIENCE_RELS") else True
    scars = ni._parse_landed_science_scars(REL, _doc(), "landed")
    assert len(scars) == 1
    assert scars[0].original_id == FAMILY
    assert scars[0].refuse_eligible is True
    assert scars[0].verdict == "MEASURED_NEGATIVE"


def test_the_mechanism_names_both_measured_rates():
    mech = _doc()["scars"][0]["mechanism"]
    assert "codes_keep_50" in mech and "aux_keep_50" in mech
    assert "0.547282" in mech and "1827.21" in mech
    assert "344.1 organ" in mech


def test_it_does_not_overclaim():
    """Auxiliary bytes are not free to STORE, and another packing might cost time."""
    nots = _doc()["scars"][0]["not"]
    assert "free to STORE" in nots
    assert "THIS packing" in nots
