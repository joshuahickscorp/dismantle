"""The agentic loop: a model that needs to LOOK at something can now say so.

Before this, HCLI_RESULT_SCHEMA was edit-only -- `operations` are file edits, so a
model with a question about the repo had no field to put a tool call in. 61
registered tools were unreachable from the natural-language path BY SCHEMA
CONSTRUCTION, and the measured symptom was HCLI answering "no deterministic
evidence provided" to "list the python files in the hcli directory".

These tests use a stub model so the loop's control flow is checked deterministically,
without a resident. The live end-to-end run is separate and slow.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcli.engine import Engine, HCLI_RESULT_SCHEMA
from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[1]


def _registry():
    return default_tool_registry(REPO, repo_root=REPO)


def test_the_schema_can_express_a_tool_call_at_all():
    """The defect was structural: there was nowhere to put one."""
    props = HCLI_RESULT_SCHEMA["properties"]
    assert "tool_calls" in props, "no field a tool call could occupy"
    assert "tool_use" in props["kind"]["enum"]
    # strict mode: every property must be required or the backend rejects it
    assert set(HCLI_RESULT_SCHEMA["required"]) == set(props)
    item = props["tool_calls"]["items"]["properties"]
    assert set(item) == {"tool", "arguments"}
    # Arguments must NOT be a JSON-encoded string. That version was tried and
    # measurably failed: the model could not escape quotes inside quotes and
    # blew all 3 structured-output attempts on "Expecting ',' delimiter".
    assert item["arguments"]["type"] == "array"
    assert item["arguments"]["items"]["properties"]["value"]["type"] == "string"


def test_string_pairs_become_the_types_each_tool_declares():
    """The model can only emit strings; tools want ints and bools."""
    registry = _registry()
    args = Engine._typed_arguments(registry, "fs.list", [
        {"name": "path", "value": "hcli"},
        {"name": "max_results", "value": "5"},
        {"name": "recursive", "value": "false"},
    ])
    assert args == {"path": "hcli", "max_results": 5, "recursive": False}
    # and the coerced call must actually satisfy the registry
    result = registry.invoke("fs.list", args)
    assert result.ok, result.error
    assert len(result.value["files"]) == 5


def test_an_uncoercible_value_reaches_the_registry_as_a_readable_error():
    """Never guess. A bad value must surface as text the model can react to."""
    registry = _registry()
    args = Engine._typed_arguments(registry, "fs.list", [
        {"name": "path", "value": "hcli"},
        {"name": "max_results", "value": "not-a-number"},
    ])
    assert args["max_results"] == "not-a-number"  # left alone, not invented
    assert registry.invoke("fs.list", args).ok is False


def test_a_failing_tool_is_an_observation_not_an_exception():
    """A daemon that dies on a bad argument is not unattended.

    Every failure mode here -- unknown tool, missing required argument, a handler
    that raises -- must come back as readable text.
    """
    engine = Engine.__new__(Engine)
    engine._tools_cached = _registry()
    engine.MAX_EVIDENCE_CHARS_PER_FILE = 4000
    engine._emit = lambda *a, **k: None

    out = engine._run_tool_calls([
        {"tool": "no.such.tool", "arguments": []},
        {"tool": "fs.search", "arguments": [{"name": "root", "value": "hcli"}]},
        {"tool": "fs.read", "arguments": [{"name": "path", "value": "does/not/exist.py"}]},
    ], goal_id="g")

    assert len(out) == 3
    assert all(o["ok"] is False for o in out)
    assert "unknown tool" in out[0]["text"]
    # THE FIX THAT MATTERS: the signature travels WITH the error, so the retry
    # does not have to guess. Without it the model burned a whole tool budget
    # calling fs.search without `pattern` and guessing again each round.
    assert "pattern" in out[1]["text"] and "signature:" in out[1]["text"]
    assert "pattern*" in out[1]["text"], "required args must be marked"


def test_the_catalog_tells_the_model_the_arguments_not_just_the_names():
    catalog = Engine._tool_catalog(_registry())
    assert "fs.list(" in catalog
    line = next(l for l in catalog.splitlines() if l.startswith("fs.search("))
    assert "pattern*:string" in line, "required marker missing"
    assert "root:string" in line and "root*" not in line, "optional marked required"


def test_a_directory_listing_verb_exists():
    """It did not, and that cost an entire tool budget on the first real query.

    61 tools and none could answer "what files are in this directory": fs.search
    requires a content `pattern`, so listing was inexpressible. The model was not
    confused, the capability was absent.
    """
    registry = _registry()
    assert registry.get("fs.list") is not None
    result = registry.invoke("fs.list", {"path": "hcli", "glob": "*.py", "recursive": False})
    assert result.ok, result.error
    names = {row["path"] for row in result.value["files"]}
    assert "engine.py" in names and "tool_registry.py" in names
    assert all(row["path"].endswith(".py") for row in result.value["files"])


def test_tool_output_never_enters_the_evidence_list():
    """Evidence items are hashed, size/mtime-stamped file snapshots that get
    re-read when they change. Tool output is none of those. Letting it into the
    same list would let unhashed, model-directed content pose as deterministic
    evidence and quietly weaken the freshness gate."""
    engine = Engine.__new__(Engine)
    engine._tools_cached = _registry()
    text = engine._prompt_with_observations(
        "the goal", [{"tool": "fs.list", "ok": True, "text": "engine.py"}]
    )
    assert "OBSERVATIONS" in text and "engine.py" in text
    assert "AVAILABLE TOOLS" in text
    assert "TOOL BUDGET EXHAUSTED" not in text
    final = engine._prompt_with_observations("the goal", [], final=True)
    assert "TOOL BUDGET EXHAUSTED" in final


def test_sanitizer_accepts_tool_use_and_drops_nameless_calls():
    engine = Engine.__new__(Engine)
    out = engine._sanitize_result({
        "kind": "tool_use", "content": "looking",
        "operations": [], "tests": [],
        "tool_calls": [
            {"tool": "fs.list", "arguments": [{"name": "path", "value": "hcli"}]},
            {"tool": "   ", "arguments": []},
            "not-a-dict",
        ],
    })
    assert out["kind"] == "tool_use"
    assert [c["tool"] for c in out["tool_calls"]] == ["fs.list"]


def test_an_answer_still_carries_no_tool_calls():
    engine = Engine.__new__(Engine)
    out = engine._sanitize_result(
        {"kind": "answer", "content": "hi", "operations": [], "tests": []}
    )
    assert out["kind"] == "answer" and out["tool_calls"] == []
