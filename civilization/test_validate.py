"""Mutation tests. A validator nobody has watched REFUSE is indistinguishable from
one that always accepts -- the check-that-cannot-fail this program has sealed six
times. Every test here breaks the ledger on purpose and requires a violation."""
import copy, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from validate import validate

HERE = pathlib.Path(__file__).resolve().parent
GOAL = (pathlib.Path.home() / ".claude/ultragoal/hawking-odyssey-maxx-ascension/GOAL.md").read_text()
def state(): return json.loads((HERE / "ROADMAP_STATE.json").read_text())


def test_the_real_ledger_is_defensible():
    assert validate(state(), GOAL) == []


def test_inflating_completion_to_evidence_coverage_is_REFUSED():
    """The exact mistake the first build made: I-D 9/9 categories, 0/8 obligations,
    printed 100%."""
    s = state(); s["civilization_status"]["I-D_ACCELERATOR"]["completion_pct"] = 100.0
    assert any("inflation" in b for b in validate(s, GOAL))


def test_COMPLETE_with_an_open_gate_is_REFUSED():
    s = state(); c = s["civilization_status"]["I-D_ACCELERATOR"]
    c["status"] = "CIVILIZATION_COMPLETE"
    assert any("open gates" in b for b in validate(s, GOAL))


def test_an_unmapped_obligation_is_REFUSED():
    s = state(); s["unmapped_obligations"] = ["G099"]
    assert any("unmapped" in b for b in validate(s, GOAL))


def test_a_status_count_disagreeing_with_GOAL_md_is_REFUSED():
    """Disk state is authority. Retyping a count here must not survive."""
    s = state(); s["obligation_status_counts"]["VERIFIED"] = 58
    assert any("disagree with GOAL.md" in b for b in validate(s, GOAL))


def test_a_fabricated_test_count_is_REFUSED():
    s = state(); s["last_verified_test_count"] = "458"          # a string, not a run
    assert any("not an integer" in b for b in validate(s, GOAL))
    s = state(); s["test_count_is_from_a_run_not_arithmetic"] = False
    assert any("not marked as coming from a run" in b for b in validate(s, GOAL))


def test_evidence_pct_must_match_its_own_table():
    s = state(); s["civilization_status"]["I-B_DOCTOR"]["evidence_pct"] = 100.0
    assert any("does not match its own category table" in b for b in validate(s, GOAL))


def test_dropping_the_named_gates_is_REFUSED():
    s = state(); s["named_gates"] = {}
    assert any("no named gates" in b for b in validate(s, GOAL))
