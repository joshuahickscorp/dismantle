"""The dashboard must not invent a zero for an instrument that did not report.

Directive XXXII: "Unknown means NOT_INSTRUMENTED. Never silently substitute
zero." ``hcli_metric`` currently sums ``prefill_profile.totals.wall_ns`` with a
default of ``0``, so a model call that carried no prefill profile contributes
nothing and reads as a call whose prefill was free. The second half is worse
than the first: decode is derived as ``wall - prefill``, so the missing prefill
is not merely lost, it is REATTRIBUTED to decode. A reader sees a unit that
spent all its wall decoding.

The contract this pins:

* ``prefill_seconds(calls)`` returns ``None`` when ANY call in the unit lacks a
  prefill measurement -- an unknown addend makes the whole sum unknown.
* ``decode_seconds(calls)`` returns ``None`` whenever prefill is unknown, because
  ``wall - unknown`` is unknown, not ``wall``.
* Both return the real number when every call is instrumented.
"""
from __future__ import annotations

import pytest

from tools import hcli_metric


def _call(wall_s: float, prefill_ns: int | None) -> dict:
    call: dict = {"wall_s": wall_s, "prompt_tokens": 10, "completion_tokens": 5}
    if prefill_ns is not None:
        call["prefill_profile"] = {"totals": {"wall_ns": prefill_ns}}
    return call


def test_fully_instrumented_unit_reports_real_seconds():
    cs = [_call(10.0, 6_000_000_000), _call(4.0, 1_000_000_000)]
    assert hcli_metric.prefill_seconds(cs) == pytest.approx(7.0)
    assert hcli_metric.decode_seconds(cs) == pytest.approx(7.0)


def test_one_uninstrumented_call_makes_the_whole_prefill_unknown():
    cs = [_call(10.0, 6_000_000_000), _call(4.0, None)]
    assert hcli_metric.prefill_seconds(cs) is None, (
        "a call with no prefill_profile contributed 0 and the sum was reported "
        "as if it were measured"
    )


def test_unknown_prefill_is_not_reattributed_to_decode():
    cs = [_call(10.0, 6_000_000_000), _call(4.0, None)]
    assert hcli_metric.decode_seconds(cs) is None, (
        "decode was computed as wall - prefill with prefill defaulted to 0, so "
        "the uninstrumented prefill was credited to decode"
    )


def test_a_unit_with_no_calls_at_all_is_unknown_not_zero():
    assert hcli_metric.prefill_seconds([]) is None
    assert hcli_metric.decode_seconds([]) is None
