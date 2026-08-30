"""Negative controls for the generic NR→NX pipeline.

A source-tree runtime read, a renamed source pointer, a skipped stage, or a
physical_ebpw value would be the campaign repeating a pass it did not earn.
These tests make each of those refusals fire. An absent specimen is a
recorded refusal, never pytest.skip.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import flash_nx_audit as nx_audit
from tools.future import nr_nx_generic as nng
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


def _receipt() -> dict:
    path = nng.build()
    return json.loads(path.read_text())


def test_build_emits_sealed_static_receipt():
    out = nng.build()
    assert out.name == "NR_NX_GENERIC.json"
    assert out.parent == RECEIPTS
    doc = json.loads(out.read_text())
    assert doc["schema"] == nng.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["physical_ebpw"] is None
    assert doc["physical_ebpw_written"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["receipt"] == "receipts/future/NR_NX_GENERIC.json"
    assert doc["resident_callable"]["frontier"] == "FT.MODEL_EXECUTION.complete-token"


def test_generic_and_flash_facts_are_separate_and_not_merged():
    doc = _receipt()
    assert "GENERIC_NR_NX_PIPELINE_CALLABLE" in doc
    assert "FLASH_NX_READY" in doc
    assert doc["facts_are_independent"] is True
    assert doc["GENERIC_NR_NX_PIPELINE_CALLABLE"] is False
    assert doc["FLASH_NX_READY"] is False
    assert doc["flash"]["FLASH_NX_READY"] is False
    # A Flash metadata seal must not be laundered into a generic pass.
    assert not (
        doc["GENERIC_NR_NX_PIPELINE_CALLABLE"] is True and doc["FLASH_NX_READY"] is False
        and doc.get("first_nx_lower_failure") is None
    )


def test_stages_are_complete_never_skipped():
    doc = _receipt()
    names = [s["stage"] for s in doc["stages"]]
    assert names == list(nng.STAGE_ORDER)
    for row in doc["stages"]:
        assert row["status"] not in nng.FORBIDDEN_STAGE_STATUS
        assert row["status"] != "SKIPPED"
        assert row["status"] in {nng.PASSED, nng.FAILED, nng.REFUSED, nng.BLOCKED}
        assert row["why"]
        assert "invoked" in row


def test_pipeline_callable_is_false_because_a_stage_did_not_pass():
    doc = _receipt()
    assert nng.generic_pipeline_callable(doc["stages"]) is False
    assert doc["GENERIC_NR_NX_PIPELINE_CALLABLE"] is False
    failed = [s for s in doc["stages"] if s["status"] != nng.PASSED]
    assert failed, "callable is False; at least one stage must not have passed"
    first = doc["first_failing_stage"]
    assert first is not None
    assert first["stage"] == failed[0]["stage"]


def test_skipped_stage_cannot_be_declared_callable():
    """NEGATIVE CONTROL: a SKIPPED stage is a constructor error, not a pass."""
    with pytest.raises(nng.StageSkipForbidden):
        nng._stage("Verifier", "SKIPPED", why="nope", invoked=False)
    fake = [
        nng._stage(name, nng.PASSED, why="synthetic all-pass", invoked=True)
        for name in nng.STAGE_ORDER
    ]
    fake[0] = dict(fake[0])
    fake[0]["status"] = "SKIPPED"
    assert nng.generic_pipeline_callable(fake) is False
    with pytest.raises(nng.StageSkipForbidden):
        nng.declare_pipeline_callable(fake, packed_nx_path=None)


def test_all_passed_without_packed_nx_still_refuses_callable(tmp_path):
    """NEGATIVE CONTROL: passing stages with no NX body is not a pipeline."""
    fake = [
        nng._stage(name, nng.PASSED, why="synthetic", invoked=True)
        for name in nng.STAGE_ORDER
    ]
    assert nng.generic_pipeline_callable(fake) is True
    with pytest.raises(nng.PipelineCallableForbidden):
        nng.declare_pipeline_callable(fake, packed_nx_path=None)
    with pytest.raises(nng.PipelineCallableForbidden):
        nng.declare_pipeline_callable(fake, packed_nx_path=tmp_path / "missing.nx")
    nx_path = tmp_path / "packed.nx.json"
    nx_path.write_text("{}")
    assert nng.declare_pipeline_callable(fake, packed_nx_path=nx_path) is True


def test_source_independence_fails_on_runtime_read_into_source_tree():
    """NEGATIVE CONTROL: any runtime read into the specimen must fail."""
    tree = "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0"
    nx = {
        "status": "SOURCE_INDEPENDENT_COMPLETE",
        "source_independent": True,
        "serialized_artifact": {
            "path": "/tmp/packed.nxbin",
            "sha256": "a" * 64,
            "status": "BUILT",
            "self_contained": True,
        },
        "physical_loader": {"status": "BUILT", "source_independent": True},
        "runtime_reads": [f"{tree}/model.safetensors"],
    }
    judged = nng.source_independence(nx, source_trees=[tree])
    assert judged["ok"] is False
    assert any("runtime_reads" in h for h in judged["hits"])
    assert "source tree" in judged["why"]


def test_source_independence_fails_on_renamed_source_pointer():
    """NEGATIVE CONTROL: labeling the parent checkpoint as the NX body is a pointer."""
    tree = "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0"
    nx = {
        "status": "SOURCE_INDEPENDENT_COMPLETE",
        "source_independent": True,
        "serialized_artifact": {
            "path": f"{tree}/model.safetensors",
            "sha256": "b" * 64,
            "status": "BUILT",
            "self_contained": True,
        },
        "physical_loader": {"status": "BUILT", "source_independent": True},
    }
    judged = nng.source_independence(nx, source_trees=[tree])
    assert judged["ok"] is False
    assert any("source tree" in h or "renamed source pointer" in h for h in judged["hits"])


def test_source_independence_can_pass_on_a_self_contained_body():
    """Inverse: the checker must still be able to return ok=True."""
    nx = nx_audit.synthetic_promotable_nx()
    judged = nng.source_independence(
        nx,
        source_trees=["/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0"],
    )
    assert judged["ok"] is True
    assert judged["hits"] == []


def test_source_independence_fails_without_an_nx():
    judged = nng.source_independence(None, source_trees=[])
    assert judged["ok"] is False
    assert "no NX" in judged["why"]


def test_flash_v0_fails_source_independence_when_present():
    """The live Flash metadata seal is a real negative, not a synthetic one."""
    path = nx_audit.evidence_path(nx_audit.REL_NX_V0)
    if path is None:
        judged = nng.source_independence(
            {"status": nx_audit.METADATA_ONLY, "source_independent": False},
            source_trees=[],
        )
        assert judged["ok"] is False
        assert "metadata" in judged["why"]
        return
    nx = json.loads(path.read_text())
    judged = nng.source_independence(nx, source_trees=[])
    assert judged["ok"] is False
    assert nx_audit._status_is_metadata_only(nx)


def test_no_code_path_writes_a_physical_ebpw_value():
    """NEGATIVE CONTROL: record_physical_ebpw always raises."""
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.record_physical_ebpw(0.887)
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.record_physical_ebpw(16.0)
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.record_physical_ebpw(None)
    doc = _receipt()
    nng.assert_no_physical_ebpw(doc)
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.assert_no_physical_ebpw({"physical_ebpw": 0.5})
    with pytest.raises(nng.PhysicalEbpwForbidden):
        nng.assert_no_physical_ebpw({"nested": {"qualified_complete_physical_ebpw": 1.0}})
    src = Path(nng.__file__).read_text()
    tree = ast.parse(src)
    assigned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in nnp_keys():
                    assigned.append(t.id)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in nnp_keys():
                assigned.append(node.target.id)
    assert assigned == []


def nnp_keys():
    return nng.nnp.PHYSICAL_EBPW_KEYS


def test_receipt_refuses_hardware_fields():
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "NR_NX_GENERIC_SHOULD_NOT_EXIST.json",
            {"schema": "x", "tps": 1.0},
            "tools/future/test_nr_nx_generic.py",
        )
    for field in HARDWARE_FIELDS:
        assert field not in {"schema", "version", "purpose"}


def test_choose_specimen_refuses_when_nothing_is_verified():
    """NEGATIVE CONTROL: the chooser can return the negative."""
    row = nng.choose_specimen(present=set(), verified={}, lake_mounted=False)
    assert row["ok"] is False
    assert row["id"] is None
    assert "Refusing to invent a specimen" in row["why"]


def test_choose_specimen_does_not_silently_substitute_falcon():
    """Falcon is verified and present; it is still not the cheap dense choice."""
    verified = {
        nng.FALCON_ID: {"status": "WHOLE_TREE_VERIFIED", "whole_tree_verified": True, "bytes_hashed": 15},
    }
    row = nng.choose_specimen(
        present={nng.FALCON_ID},
        verified=verified,
        lake_mounted=True,
    )
    assert row["ok"] is False
    assert "Falcon is not substituted" in row["why"]


def test_choose_specimen_selects_qwen06_when_verified():
    verified = {
        nng.QWEN06_ID: {
            "status": "WHOLE_TREE_VERIFIED",
            "whole_tree_verified": True,
            "bytes_hashed": 1519209243,
            "n_files": 10,
            "specimen_path": "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0",
        },
        nng.FALCON_ID: {"status": "WHOLE_TREE_VERIFIED", "whole_tree_verified": True},
    }
    row = nng.choose_specimen(
        present={nng.QWEN06_ID, nng.FALCON_ID},
        verified=verified,
        lake_mounted=True,
    )
    assert row["ok"] is True
    assert row["id"] == nng.QWEN06_ID
    assert "Falcon" in row["why_not_falcon"]


def test_native_engine_parser_can_accept_and_reject():
    """NEGATIVE CONTROL: the allowlist parser returns both a hit and a miss."""
    native = nng.native_engine_architectures()
    if not native.get("ok"):
        stub = '''
        match arch.as_str() {
            "qwen2" | "qwen2.5" | "qwen" => {}
            "qwen2moe" | "qwen3moe" | "qwen-moe" => {}
            other => Err(Error::Model(format!("unknown architecture")))
        }
        '''
        native = nng.native_engine_architectures(stub)
    assert native["ok"] is True
    assert native["includes_qwen2"] is True
    assert native["includes_qwen3moe"] is True
    assert native["includes_qwen3_dense"] is False
    assert native["includes_falcon_h1"] is False
    assert "qwen3" not in (native.get("architectures") or [])


def test_physical_graph_compiler_failure_is_named_when_specimen_is_present():
    doc = _receipt()
    choice = doc["specimen"]
    pgc = next(s for s in doc["stages"] if s["stage"] == "PhysicalGraphCompiler")
    if choice.get("ok") and Path(str(choice.get("specimen_path") or "")).is_dir():
        assert pgc["status"] == nng.FAILED
        assert pgc["invoked"] is True
        assert pgc["error"]
        assert "model.safetensors.index.json" in (pgc.get("error") or "") or "index.json" in pgc["why"]
        assert doc["first_nx_lower_failure"]["stage"] == "PhysicalGraphCompiler"
        evidence = pgc["evidence"]
        assert evidence["collapse_tensors_absent"]
        assert evidence["emit_written"] is False
        assert evidence["returncode"] != 0
    else:
        assert pgc["status"] == nng.REFUSED
        assert "not on disk" in pgc["why"] or "not" in pgc["why"].lower()


def test_sleeping_unit_is_sleeping_never_pending():
    doc = _receipt()
    wu = doc["sleeping_workunit"]
    assert wu["status"] == "sleeping"
    assert wu["classification"] == "SLEEPING"
    assert wu["status"] not in {"pending", "PENDING", "ready", "READY"}
    assert wu["synthetic_result_forbidden"] is True
    assert wu["wake_unmet"]
    assert any(not w["holds"] for w in wu["wake_conditions"])


def test_check_nx_verifier_was_invoked():
    doc = _receipt()
    ver = next(s for s in doc["stages"] if s["stage"] == "Verifier")
    assert ver["invoked"] is True
    assert ver["status"] != nng.PASSED
    assert ver["evidence"]["promotable"] is False


def test_architecture_recognizer_ran_or_refused_without_skipping():
    doc = _receipt()
    row = next(s for s in doc["stages"] if s["stage"] == "ArchitectureRecognizer")
    if doc["specimen"].get("ok") and Path(str(doc["specimen"].get("specimen_path") or "")).is_dir():
        assert row["status"] == nng.PASSED
        assert row["invoked"] is True
        assert row["evidence"]["loaded_weights"] is False
        organs = [o["organ"] for o in row["evidence"]["organs"]]
        assert "mlp_gate_up" in organs
        assert "gqa_attention" in organs
        assert row["evidence"]["n_unmatched"] == 0
        assert row["evidence"]["did_not_fetch_network"] is True
    else:
        assert row["status"] == nng.REFUSED
        assert row["status"] != "SKIPPED"


def test_no_pytest_skip_in_this_file():
    src = Path(__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            assert name != "skip", "pytest.skip that actually fires is a P0"
