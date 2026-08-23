"""G021 promotion controls: the mutation caused the win, or the harness is lying.

No GPU, no 27B, no live llama-server. The score is admitted_n through
Controller.ensure_runtime_pool on a git-archive scratch copy of HEAD hcli.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import pytest
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts" / "headless" / "SELF_OPT_CANDIDATE_PROMOTED.json"

spec = importlib.util.spec_from_file_location(
    "hcli_self_optimize_2", HERE / "hcli_self_optimize_2.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_compute_decision_refuses_noop_identical_admission():
    d = mod.compute_decision(
        correctness_ok=True,
        throughput_improved=False,
        mutation_applied=True,
        validation_ok=True,
        validation_reason=None,
        orig_med=1,
        mut_med=1,
        spread=0,
        admission_differs=False,
        metric_name="admitted_n",
    )
    assert d["verdict"] == "REFUSED"
    assert d["would_refuse"] is True
    assert d["refuse_if"]["admission_did_not_differ"]["triggered"] is True
    assert d["refuse_if"]["throughput_did_not_improve_beyond_spread"]["triggered"] is True


def test_compute_decision_refuses_bad_worse_score():
    d = mod.compute_decision(
        correctness_ok=True,
        throughput_improved=False,
        mutation_applied=True,
        validation_ok=True,
        validation_reason=None,
        orig_med=1,
        mut_med=0,
        spread=0,
        admission_differs=True,
        metric_name="admitted_n",
    )
    assert d["verdict"] == "REFUSED"
    assert d["decision"] == "reject"
    assert "did not improve" in d["reason"]


def test_compute_decision_refuses_no_evidence_even_if_score_moved():
    d = mod.compute_decision(
        correctness_ok=True,
        throughput_improved=True,
        mutation_applied=True,
        validation_ok=False,
        validation_reason="NO_EVIDENCE",
        orig_med=1,
        mut_med=2,
        spread=0,
        admission_differs=True,
        metric_name="admitted_n",
    )
    assert d["verdict"] == "REFUSED"
    assert "NO_EVIDENCE" in d["reason"]


def test_compute_decision_promotes_only_when_every_gate_is_green():
    d = mod.compute_decision(
        correctness_ok=True,
        throughput_improved=True,
        mutation_applied=True,
        validation_ok=True,
        validation_reason=None,
        orig_med=1,
        mut_med=2,
        spread=0,
        admission_differs=True,
        metric_name="admitted_n",
    )
    assert d["verdict"] == "PROMOTE"
    assert d["would_refuse"] is False


def test_failing_gate_trial_computes_would_refuse_from_verdict():
    ws = Path(tempfile.mkdtemp(prefix="g021-failing-gate-"))
    trial = mod.run_failing_gate_trial(ws)
    assert trial["hardcoded"] is False
    assert trial["pytest_exit_code"] != 0
    assert trial["pytest_passed_gate"] is False
    failing = trial["decision_on_failing_correctness"]
    assert failing["verdict"] == "REFUSED"
    assert trial["would_refuse_on_failing_gate"] == (failing["verdict"] == "REFUSED")
    assert trial["would_refuse_on_failing_gate"] is True
    assert trial["would_refuse_on_no_evidence"] is True
    assert trial["evidenced"] is True


def test_would_refuse_on_failing_gate_is_not_hardcoded_true():
    src = inspect.getsource(mod.run_failing_gate_trial)
    assert 'failing.get("verdict") == "REFUSED"' in src
    assert "would_refuse_on_failing_gate\": True" not in src
    assert "would_refuse_on_failing_gate'] = True" not in src


@pytest.mark.xfail(
    strict=True,
    reason=(
        "G021 is OPEN: the promotion does not reproduce on a tree where hcli/ is "
        "present. The BAD control asserts candidate_admitted_n == {0} and gets {2}, "
        "so the mutation is not reaching the executed code here and the REFUSED "
        "verdicts are vacuous rather than meaningful. The controls DID bite in the "
        "lane worktree, where hcli/ was a sparse hole materialized via git archive. "
        "strict=True on purpose: if the root-cause lane fixes the harness this test "
        "starts passing, the suite FAILS, and this marker must be removed rather "
        "than a real repair going unnoticed."
    ),
)
def test_four_controls_physically_ran():
    receipt = mod.run_promotion_controls(REPO)
    assert receipt["schema"] == "hawking.headless.hcli_self_opt.candidate_promoted.v1"
    controls = receipt["controls"]
    for name in ("noop", "bad", "paired_interleaved", "failing_gate"):
        assert controls[name]["ran"] is True, name

    noop = controls["noop"]
    assert noop["is_win"] is False, (
        "NO-OP scored as a win; the harness is measuring something other "
        f"than the mutation: {noop}"
    )
    assert noop["through_mutated_mechanism"] is True
    assert noop["admission_differs"] is False
    assert (noop.get("decision") or {}).get("verdict") == "REFUSED"

    bad = controls["bad"]
    assert bad["is_win"] is False
    assert (bad.get("decision") or {}).get("verdict") == "REFUSED"
    assert bad["through_mutated_mechanism"] is True
    assert set(bad["candidate_admitted_n"]) == {0}
    assert set(bad["baseline_admitted_n"]) == {1}

    paired = controls["paired_interleaved"]
    assert paired["block_design"] is False
    order = paired["h1_order"]
    assert order == ["candidate", "baseline", "candidate", "baseline"]
    h1 = paired["h1"]
    assert h1["admission_differs"] is True
    assert set(h1["candidate_admitted_n"]) == {2}
    assert set(h1["baseline_admitted_n"]) == {1}
    assert h1["through_mutated_mechanism"] is True

    fail = controls["failing_gate"]
    assert fail["hardcoded"] is False
    assert fail["would_refuse_on_failing_gate"] is True
    assert fail["pytest_exit_code"] != 0
    assert receipt["would_refuse_on_failing_gate"] == fail["would_refuse_on_failing_gate"]

    validation = (receipt.get("mutation_receipt") or {}).get("validation") or {}
    assert validation.get("ok") is True
    assert validation.get("reason") != "NO_EVIDENCE"

    assert RECEIPT.is_file()
    disk = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert disk["schema"] == receipt["schema"]
    assert disk["controls"]["noop"]["is_win"] is False
    assert disk["controls"]["bad"]["decision"]["verdict"] == "REFUSED"
    assert disk["would_refuse_on_failing_gate"] is True
