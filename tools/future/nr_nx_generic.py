"""NR_NX_GENERIC — prove the generic NR→NX pipeline on a real small specimen.

nr_nx_path_callable currently fails because Flash has no packed final NX.
That conflates a generic orchestration capability with one model's artifact
readiness, and it contradicts the bootstrap in which Qwen27 launches Odyssey
while Flash remains an evolving child.

This module drives the real compiler stages on the cheapest whole-tree-verified
specimen the compiler can actually see. It does not pack an NX, does not mint a
fixture, does not rename a source pointer, does not write physical EBPW, and
does not declare the pipeline callable when any stage was skipped or refused.

    python3 tools/future/nr_nx_generic.py --build
    python3 -m pytest tools/future/test_nr_nx_generic.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import RECEIPTS, REPO, git, load_json, write_receipt
from tools.future import flash_nx_audit as nx_audit
from tools.future import nr_nx_path as nnp
from tools.future import specimen_verify as sv
from tools.future import workunit_species as wus
from tools.odyssey import arch_recognizer as ar
from tools.odyssey import doctor_tournament as doctor
from tools.odyssey import noetic_compiler as nc
from tools.odyssey import physical_graph_compiler as pgc

RECEIPT = "NR_NX_GENERIC.json"
SCHEMA = "hawking.future.nr_nx_generic.v1"
RECORDED_BY = "tools/future/nr_nx_generic.py"
VERSION = 1

REL_VERIFY = "receipts/future/SPECIMEN_VERIFICATION.json"
REL_MATRIX = "receipts/headless/ORGAN_FRONTIER_MATRIX.json"
REL_KERNELS = "receipts/headless/KERNEL_LIBRARY.json"
REL_AUDIT = "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json"
NATIVE_LOADER = "crates/hawking-core/src/model/mod.rs"
HEADLESS_FIRST_NX = "tools/headless/first_noetic_executable.py"

PASSED = "PASSED"
FAILED = "FAILED"
REFUSED = "REFUSED"
BLOCKED = "BLOCKED"
SLEEPING = "SLEEPING"

# A skipped stage is how a pipeline pretends to finish. The constructor refuses it.
FORBIDDEN_STAGE_STATUS = frozenset({"SKIPPED", "skip", "pending", "PENDING", "READY", "ready"})

STAGE_ORDER: tuple[str, ...] = (
    "SpecimenSelect",
    "SpecimenPresent",
    "ArchitectureRecognizer",
    "OrganGraph",
    "NrIdentifyOrCreate",
    "Doctor",
    "RepresentationPlanner",
    "PhysicalGraphCompiler",
    "KernelPlanner",
    "DeviceCompiler",
    "NoeticExecutable",
    "SourceIndependence",
    "ExecutableDependencyAccounting",
    "Verifier",
)

QWEN06_ID = "Qwen--Qwen3-0.6B@c1899de289a0"
QWEN06_REPO = "Qwen/Qwen3-0.6B"
QWEN06_REV = "c1899de289a0"
FALCON_ID = "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb"
FALCON_REPO = "tiiuae/Falcon-H1-7B-Instruct"
FALCON_REV = "41e72f27effb"

# Collapse tensors PhysicalGraphCompiler.get() will look up. Dense Qwen3-0.6B
# stores the same roles without `.experts.N.`; Falcon uses feed_forward.* and
# mamba.*. Either way the compiler's hardcoded MoE path is a miss.
PGC_COLLAPSE_TENSORS: tuple[str, ...] = (
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
    "model.layers.0.mlp.experts.0.down_proj.weight",
    "model.layers.0.mlp.gate.weight",
)

SOURCE_FILE_NAMES = frozenset(
    {
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "consolidated.safetensors",
    }
)


class StageSkipForbidden(ValueError):
    """A stage reported SKIPPED. That is how a pipeline pretends to finish."""


class PipelineCallableForbidden(ValueError):
    """Callable was claimed while a stage was not PASSED or an NX was missing."""


PhysicalEbpwForbidden = nnp.PhysicalEbpwForbidden


def record_physical_ebpw(value: Any) -> None:
    nnp.record_physical_ebpw(value)


def assert_no_physical_ebpw(doc: Mapping[str, Any]) -> None:
    nnp.assert_no_physical_ebpw(doc)


def _dot(node: Any, dotted: str, default: Any = None) -> Any:
    cur: Any = node
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _stage(
    name: str,
    status: str,
    *,
    why: str,
    invoked: bool,
    evidence: Any = None,
    error: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status in FORBIDDEN_STAGE_STATUS or status == "SKIPPED":
        raise StageSkipForbidden(
            f"{name}: status={status!r} is forbidden; a stage that cannot run "
            "is FAILED, REFUSED, or BLOCKED with a reason"
        )
    if status not in {PASSED, FAILED, REFUSED, BLOCKED}:
        raise StageSkipForbidden(f"{name}: unknown status {status!r}")
    row: dict[str, Any] = {
        "stage": name,
        "status": status,
        "why": why,
        "invoked": invoked,
        "error": error,
        "evidence": evidence,
    }
    if extra:
        row.update(dict(extra))
    return row


def generic_pipeline_callable(stages: Sequence[Mapping[str, Any]]) -> bool:
    """True only when every named stage ran and PASSED. Empty is not success."""
    if not stages:
        return False
    names = [s.get("stage") for s in stages]
    if list(names) != list(STAGE_ORDER):
        return False
    for row in stages:
        if row.get("status") in FORBIDDEN_STAGE_STATUS or row.get("status") == "SKIPPED":
            return False
        if row.get("status") != PASSED:
            return False
        if row.get("invoked") is not True:
            return False
    return True


def declare_pipeline_callable(
    stages: Sequence[Mapping[str, Any]],
    *,
    packed_nx_path: str | Path | None,
) -> bool:
    """Refuse the Goodhart: SKIPPED stages or a missing NX cannot be a pass."""
    for row in stages:
        st = row.get("status")
        if st in FORBIDDEN_STAGE_STATUS or st == "SKIPPED":
            raise StageSkipForbidden(
                f"{row.get('stage')}: skipped/pending is not a result"
            )
    ok = generic_pipeline_callable(stages)
    if not ok:
        return False
    path = Path(packed_nx_path) if packed_nx_path else None
    if path is None or not path.is_file():
        raise PipelineCallableForbidden(
            "GENERIC_NR_NX_PIPELINE_CALLABLE cannot be True without a packed "
            "NX artifact on disk"
        )
    return True


# ---------------------------------------------------------------------------
# Source independence. An NX that still opens the checkpoint has not lowered.
# ---------------------------------------------------------------------------


def _resolves_into_source(value: Any, source_trees: Sequence[str | Path]) -> bool:
    if not isinstance(value, str) or not value:
        return False
    text = value
    for tree in source_trees:
        root = str(tree)
        if not root:
            continue
        if text == root or text.startswith(root.rstrip("/") + "/") or root in text:
            return True
        try:
            Path(text).resolve().relative_to(Path(root).resolve())
            return True
        except (OSError, ValueError):
            continue
    name = Path(text).name
    if name in SOURCE_FILE_NAMES and ("specimens" in text or "modellake" in text or "partial/" in text):
        return True
    return False


def _runtime_strings(nx: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Fields a loader would actually open, not provenance citations on a receipt."""
    out: list[tuple[str, str]] = []
    art = nx_audit._serialized_artifact(nx)
    if isinstance(art, dict) and isinstance(art.get("path"), str):
        out.append(("serialized_artifact.path", art["path"]))
    elif isinstance(art, str):
        out.append(("serialized_artifact", art))
    for key in (
        "runtime_reads",
        "source_path",
        "checkpoint",
        "model_dir",
        "weights",
        "specimen_path",
        "parent_path",
    ):
        raw = nx.get(key)
        if isinstance(raw, str):
            out.append((key, raw))
        elif isinstance(raw, list):
            for i, item in enumerate(raw):
                if isinstance(item, str):
                    out.append((f"{key}[{i}]", item))
    loader = nx_audit._loader(nx)
    if isinstance(loader, Mapping):
        for key in ("path", "source_path", "reads", "checkpoint"):
            raw = loader.get(key)
            if isinstance(raw, str):
                out.append((f"physical_loader.{key}", raw))
            elif isinstance(raw, list):
                for i, item in enumerate(raw):
                    if isinstance(item, str):
                        out.append((f"physical_loader.{key}[{i}]", item))
    pp = nx.get("physical_program")
    if isinstance(pp, Mapping):
        for key in ("source_path", "checkpoint", "weights", "executor_path"):
            raw = pp.get(key)
            if isinstance(raw, str):
                out.append((f"physical_program.{key}", raw))
    return out


def source_independence(
    nx: Mapping[str, Any] | None,
    *,
    source_trees: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """FAIL if the NX would read the source tree, or if its bytes are the parent.

    A metadata seal, a missing body, or a path into the specimen is not
    independence. source_independent=True on a document that still points at
    model.safetensors is a renamed source pointer.
    """
    if not isinstance(nx, Mapping):
        return {
            "ok": False,
            "why": "no NX document; absence is not source independence",
            "hits": [],
        }
    hits: list[str] = []
    if nx_audit._status_is_metadata_only(nx):
        hits.append(f"status={nx.get('status')!r} is a metadata seal, not a packed body")
    if nx.get("source_independent") is not True:
        hits.append(f"source_independent={nx.get('source_independent')!r} is not True")
    art = nx_audit._serialized_artifact(nx)
    if isinstance(art, dict):
        if art.get("self_contained") is not True:
            hits.append("serialized_artifact.self_contained is not True")
        if not (art.get("sha256") or art.get("digest")):
            hits.append("serialized_artifact has no digest; a path without a digest is a pointer")
        if art.get("status") in {None, "NOT_BUILT", "ABSENT"}:
            hits.append(f"serialized_artifact.status={art.get('status')!r}")
    elif art in (None, "", "NOT_BUILT"):
        hits.append("no serialized_artifact; there is nothing to be independent of the source")
    loader = nx_audit._loader(nx)
    if isinstance(loader, Mapping) and loader.get("source_independent") is not True:
        hits.append(f"physical_loader.source_independent={loader.get('source_independent')!r}")
    binding = _dot(nx, "physical_program.source_binding")
    if isinstance(binding, str) and binding.strip():
        lowered = binding.lower()
        if any(tok in lowered for tok in ("source", "checkpoint", "safetensor", "specimen", "executor")):
            hits.append(f"physical_program.source_binding still names a source executor: {binding[:160]}")
    for field, value in _runtime_strings(nx):
        if _resolves_into_source(value, source_trees):
            hits.append(f"{field} resolves into the source tree: {value}")
        if Path(str(value)).name in SOURCE_FILE_NAMES and field.startswith("serialized_artifact"):
            hits.append(
                f"renamed source pointer: {field} names a source checkpoint file ({Path(str(value)).name})"
            )
    ok = not hits
    return {
        "ok": ok,
        "why": "source-independent packed body" if ok else "; ".join(hits),
        "hits": hits,
        "source_independent_flag": nx.get("source_independent"),
        "status": nx.get("status"),
    }


# ---------------------------------------------------------------------------
# Specimen choice. Cheapest whole-tree-verified dense body the compiler sees.
# ---------------------------------------------------------------------------


def _verification_index() -> dict[str, Any]:
    path = nx_audit.evidence_path(REL_VERIFY)
    if path is None:
        return {"present": False, "via": "missing", "rows": {}, "whole_tree": []}
    doc = load_json(path)
    rows: dict[str, dict[str, Any]] = {}
    for row in doc.get("results") or []:
        if isinstance(row, dict) and row.get("specimen"):
            rows[str(row["specimen"])] = row
    return {
        "present": True,
        "via": str(path),
        "rows": rows,
        "whole_tree": list(doc.get("whole_tree_verified_specimens") or []),
    }


def _safetensors_names(path: Path) -> list[str]:
    """Header-only. Payload is never mapped; independence of weights is the point."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return sorted(k for k in header if k != "__metadata__")


def _tensor_names(spec_dir: Path) -> tuple[list[str], str]:
    index = spec_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map") or {}
        return sorted(weight_map), "model.safetensors.index.json"
    shard = spec_dir / "model.safetensors"
    if shard.is_file():
        return _safetensors_names(shard), "model.safetensors header"
    return [], "absent"


def choose_specimen(
    *,
    present: set[str] | None = None,
    verified: Mapping[str, Mapping[str, Any]] | None = None,
    lake_mounted: bool | None = None,
) -> dict[str, Any]:
    """Qwen3-0.6B if it is here and whole-tree verified; Falcon is not a silent substitute.

    Falcon-H1 is in the recognizer's O001 blind set, but it is 15GB, hybrid
    recurrent_state, and the shipping engine has no falcon_h1 match arm.
    Substituting it when 0.6B is missing would hide the cheap-specimen question.
    """
    mounted = sv.available()["mounted"] if lake_mounted is None else lake_mounted
    vidx = _verification_index() if verified is None else {
        "present": True,
        "via": "caller",
        "rows": dict(verified),
        "whole_tree": [k for k, r in verified.items() if r.get("whole_tree_verified") or r.get("status") == "WHOLE_TREE_VERIFIED"],
    }
    if present is None:
        present = set()
        if mounted:
            try:
                present.update(sv.list_specimens())
            except OSError:
                present = set()
            for name in (QWEN06_ID, FALCON_ID):
                try:
                    if sv.specimen_dir(name).is_dir():
                        present.add(name)
                except Exception:
                    continue

    q06_row = (vidx.get("rows") or {}).get(QWEN06_ID) if isinstance(vidx.get("rows"), dict) else None
    falcon_row = (vidx.get("rows") or {}).get(FALCON_ID) if isinstance(vidx.get("rows"), dict) else None
    q06_verified = isinstance(q06_row, Mapping) and (
        q06_row.get("whole_tree_verified") is True or q06_row.get("status") == "WHOLE_TREE_VERIFIED"
    )
    falcon_verified = isinstance(falcon_row, Mapping) and (
        falcon_row.get("whole_tree_verified") is True or falcon_row.get("status") == "WHOLE_TREE_VERIFIED"
    )
    q06_present = QWEN06_ID in present
    falcon_present = FALCON_ID in present

    why_not_falcon = (
        "Falcon-H1-7B is whole-tree verified (~15.18GB, 751 tensors, recurrent_state) "
        "and sits in ArchitectureRecognizer's O001 blind set, but the shipping engine "
        "match arm has no falcon_h1/mamba2, PhysicalGraphCompiler is MoE-hardcoded, "
        "and a procedure question does not need a 10× heavier hybrid"
    )
    if q06_present and q06_verified:
        spec_dir = str(sv.specimen_dir(QWEN06_ID)) if mounted else (q06_row or {}).get("specimen_path")
        return {
            "ok": True,
            "id": QWEN06_ID,
            "repo": QWEN06_REPO,
            "revision": QWEN06_REV,
            "family": "dense_transformer",
            "architectures_expected": ["Qwen3ForCausalLM"],
            "specimen_path": spec_dir,
            "bytes_hashed": (q06_row or {}).get("bytes_hashed"),
            "n_files": (q06_row or {}).get("n_files"),
            "verification_status": (q06_row or {}).get("status"),
            "verification_via": vidx.get("via"),
            "why_chosen": (
                "cheapest whole-tree-verified specimen the ArchitectureRecognizer "
                "can run on without loading weights: dense Qwen3, 28 layers, "
                "single 1.50GB shard, 311 tensors. Closest shipping runtime is "
                "qwen_dense (GGUF qwen2/qwen), which is still not a qwen3 match arm"
            ),
            "why_not_falcon": why_not_falcon,
            "falcon_present": falcon_present,
            "falcon_verified": falcon_verified,
            "lake_mounted": mounted,
        }
    return {
        "ok": False,
        "id": None,
        "why": (
            f"Qwen3-0.6B present={q06_present} whole_tree_verified={q06_verified}; "
            f"Falcon present={falcon_present} whole_tree_verified={falcon_verified}; "
            f"lake_mounted={mounted}; verification_receipt={vidx.get('via')}. "
            "Refusing to invent a specimen. Falcon is not substituted: " + why_not_falcon
        ),
        "q06_present": q06_present,
        "q06_verified": q06_verified,
        "falcon_present": falcon_present,
        "falcon_verified": falcon_verified,
        "lake_mounted": mounted,
        "verification_via": vidx.get("via"),
        "why_not_falcon": why_not_falcon,
    }


def organ_library() -> dict[str, Any]:
    path = nx_audit.evidence_path(REL_MATRIX)
    if path is None:
        return {"present": False, "known": set(), "declared": set(), "via": "missing"}
    doc = load_json(path)
    rows = [e for e in (doc.get("organs") or []) if isinstance(e, dict) and e.get("organ")]
    known = {str(e["organ"]) for e in rows if e.get("status") == "MEASURED"}
    declared = {str(e["organ"]) for e in rows} - known
    return {
        "present": True,
        "known": known,
        "declared": declared,
        "via": str(path),
        "n_organs": len(rows),
    }


def native_engine_architectures(src: str | None = None) -> dict[str, Any]:
    """What crates/hawking-core actually dispatches. Naming a family is not a load."""
    path = REPO / NATIVE_LOADER
    if src is None:
        if not path.is_file():
            blob = git("show", f"HEAD:{NATIVE_LOADER}")
            if not blob:
                return {
                    "ok": False,
                    "path": NATIVE_LOADER,
                    "why": "native loader source is not on disk and git show returned empty",
                    "architectures": [],
                }
            src = blob
        else:
            src = path.read_text()
    start = src.find("match arch.as_str()")
    if start < 0:
        return {
            "ok": False,
            "path": NATIVE_LOADER,
            "why": "match arch.as_str() not found; refusing to guess the allowlist",
            "architectures": [],
        }
    chunk = src[start : start + 2500]
    quoted = re.findall(r'"([a-z0-9._-]+)"', chunk)
    other = None
    m = re.search(r'unknown architecture.*?(?:supports|engine supports)\s+([^.\\]+)', chunk, re.I | re.S)
    if m:
        other = m.group(1).strip()
    return {
        "ok": True,
        "path": NATIVE_LOADER,
        "architectures": quoted,
        "unknown_architecture_message": other,
        "includes_qwen3_dense": "qwen3" in quoted,
        "includes_qwen2": "qwen2" in quoted,
        "includes_qwen3moe": "qwen3moe" in quoted,
        "includes_falcon_h1": "falcon_h1" in quoted or "falconh1" in quoted,
        "chunk": chunk[:900],
    }


def _native_includes_qwen3_dense(native: Mapping[str, Any]) -> bool:
    arches = list(native.get("architectures") or [])
    # qwen3moe is a different family. A lone "qwen3" token would admit dense.
    return "qwen3" in arches


# ---------------------------------------------------------------------------
# Stage drivers. Each one either invokes a real entry or records the refusal.
# ---------------------------------------------------------------------------


def stage_specimen_select(choice: Mapping[str, Any]) -> dict[str, Any]:
    if choice.get("ok") is True and choice.get("id"):
        return _stage(
            "SpecimenSelect",
            PASSED,
            why=str(choice.get("why_chosen")),
            invoked=True,
            evidence={
                "id": choice.get("id"),
                "repo": choice.get("repo"),
                "bytes_hashed": choice.get("bytes_hashed"),
                "why_not_falcon": choice.get("why_not_falcon"),
            },
        )
    return _stage(
        "SpecimenSelect",
        REFUSED,
        why=str(choice.get("why") or "no specimen chosen"),
        invoked=True,
        evidence=dict(choice),
    )


def stage_specimen_present(choice: Mapping[str, Any]) -> dict[str, Any]:
    if choice.get("ok") is not True:
        return _stage(
            "SpecimenPresent",
            REFUSED,
            why="no specimen was selected; presence is not assumed",
            invoked=True,
        )
    path = Path(str(choice.get("specimen_path") or ""))
    if not path.is_dir():
        return _stage(
            "SpecimenPresent",
            REFUSED,
            why=f"specimen directory is not on disk: {path}",
            invoked=True,
            error="not_a_directory",
            evidence={"path": str(path)},
        )
    cfg = path / "config.json"
    weights = path / "model.safetensors"
    index = path / "model.safetensors.index.json"
    if not cfg.is_file():
        return _stage(
            "SpecimenPresent",
            REFUSED,
            why=f"config.json missing under {path}",
            invoked=True,
            error="missing_config",
        )
    size = weights.stat().st_size if weights.is_file() else None
    expected = None
    vrow = (_verification_index().get("rows") or {}).get(choice["id"])
    if isinstance(vrow, Mapping):
        for f in vrow.get("files") or []:
            if isinstance(f, Mapping) and f.get("file") == "model.safetensors":
                expected = f.get("bytes")
    if expected is not None and size is not None and size != expected:
        return _stage(
            "SpecimenPresent",
            FAILED,
            why=(
                f"model.safetensors size {size} != verification receipt {expected}; "
                "refusing to proceed on a drifted body"
            ),
            invoked=True,
            error="size_drift",
            evidence={"path": str(path), "size": size, "expected": expected},
        )
    return _stage(
        "SpecimenPresent",
        PASSED,
        why="specimen directory, config.json, and weight shard/index are on disk",
        invoked=True,
        evidence={
            "path": str(path),
            "config": True,
            "single_shard": weights.is_file(),
            "index_json": index.is_file(),
            "model_safetensors_bytes": size,
        },
    )


def stage_architecture_recognizer(
    choice: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any] | None,
    names: Sequence[str],
    names_via: str,
) -> dict[str, Any]:
    if choice.get("ok") is not True or cfg is None:
        return _stage(
            "ArchitectureRecognizer",
            REFUSED,
            why="no specimen/config; recognizer was not fed a default",
            invoked=False,
        )
    lib = organ_library()
    as_compiler = ar.recognize(
        str(choice["repo"]), str(choice["revision"]), dict(cfg), list(names)
    )
    organs, unknown, n_un, folded = ar.classify(
        list(names), dict(cfg), lib["known"], lib["declared"]
    )
    compiler_known_empty = not ar.known_organs()[0] and not ar.known_organs()[1]
    return _stage(
        "ArchitectureRecognizer",
        PASSED,
        why=(
            f"invoked tools.odyssey.arch_recognizer.recognize on local config+tensor "
            f"names ({names_via}); {len(organs)} organs, unmatched={n_un}; "
            "weights were not loaded"
        ),
        invoked=True,
        extra={"loaded_weights": False},
        evidence={
            "repo": choice.get("repo"),
            "revision": choice.get("revision"),
            "architectures": as_compiler.get("architectures"),
            "model_type": as_compiler.get("model_type"),
            "n_tensors": as_compiler.get("n_tensors"),
            "n_unmatched": n_un,
            "names_via": names_via,
            "organs": organs,
            "unrecognized": unknown,
            "folded_organ": folded,
            "library_via": lib.get("via"),
            "library_present": lib.get("present"),
            "compiler_known_organs_empty_in_this_checkout": compiler_known_empty,
            "compiler_as_invoked_novelty": as_compiler.get("novelty"),
            "did_not_fetch_network": True,
            "loaded_weights": False,
        },
    )


def stage_organ_graph(
    arch_row: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any] | None,
    names: Sequence[str],
) -> dict[str, Any]:
    if arch_row.get("status") != PASSED or cfg is None:
        return _stage(
            "OrganGraph",
            REFUSED,
            why="ArchitectureRecognizer did not pass; organ graph is not invented",
            invoked=False,
        )
    og = pgc.organ_graph(dict(cfg), list(names))
    return _stage(
        "OrganGraph",
        PASSED,
        why="invoked tools.odyssey.physical_graph_compiler.organ_graph (CPU, no weight load)",
        invoked=True,
        evidence=og,
    )


def stage_nr_identify(
    choice: Mapping[str, Any],
    arch_row: Mapping[str, Any],
    og_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Composition NR from organs is not a packed NR information payload.

    flash_nr_complete already refuses to bill composition-document bytes as
    serialized_nr_information. This stage identifies the organ inventory and
    records that no packed NR for this specimen exists.
    """
    if og_row.get("status") != PASSED:
        return _stage(
            "NrIdentifyOrCreate",
            REFUSED,
            why="no organ graph; a composition NR is not fabricated from nothing",
            invoked=False,
        )
    organs = list((og_row.get("evidence") or {}).get("nodes") or [])
    flash_nr = nx_audit.evidence_path(nx_audit.REL_NR_V2)
    flash_status = None
    if flash_nr is not None:
        flash_status = load_json(flash_nr).get("status")
    slots = []
    for node in organs:
        if not isinstance(node, Mapping):
            continue
        slots.append(
            {
                "organ": node.get("organ"),
                "n_tensors": node.get("n_tensors"),
                "occupying": {
                    "kind": "EXACT_CONTROL_FALLBACK",
                    "representation": "source_bf16_exact",
                    "science_mark": "COMPILE_TIME_SCIENCE_ONLY",
                },
                "packed_nr_information": False,
            }
        )
    return _stage(
        "NrIdentifyOrCreate",
        PASSED,
        why=(
            "identified a composition organ inventory for this specimen. Packed "
            "NR information payload is NOT_BUILT; composition bytes are not "
            "serialized_nr_information. Flash NR V2 is a different model"
        ),
        invoked=True,
        evidence={
            "specimen": choice.get("id"),
            "kind": "COMPOSITION_ORGAN_INVENTORY",
            "packed": False,
            "organ_count": len(slots),
            "organs": slots,
            "flash_nr_v2_path": None if flash_nr is None else str(flash_nr),
            "flash_nr_v2_status": flash_status,
            "flash_nr_is_this_specimen": False,
            "claim_boundary": (
                "exact-control occupying per organ. COMPILE_TIME_SCIENCE_ONLY. "
                "Not a packed NR. Not physical EBPW"
            ),
        },
    )


def stage_doctor(choice: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    probe_names = [
        pat.format(L=L)
        for _organ, pat, layers in doctor.PROBE_TENSORS
        for L in layers
    ]
    missing = [n for n in probe_names if n not in set(names)]
    parent = Path(doctor.PARENT)
    return _stage(
        "Doctor",
        FAILED,
        why=(
            "tools.odyssey.doctor_tournament is parameterized on a hardcoded PARENT "
            f"({parent}) with Qwen3.8-27B tensor names; {len(missing)}/{len(probe_names)} "
            "probe tensors are absent from this specimen. probes() was not run: it "
            "imports torch+safetensors and would load the wrong model"
        ),
        invoked=True,
        error="parameterized_on_wrong_parent",
        evidence={
            "entry_point": "tools/odyssey/doctor_tournament.py",
            "hardcoded_parent": str(parent),
            "parent_is_this_specimen": str(parent) == str(choice.get("specimen_path")),
            "probe_tensors": probe_names,
            "probe_tensors_missing_from_specimen": missing,
            "did_not_call_probes": True,
            "did_not_load_weights": True,
        },
    )


def stage_representation_planner(choice: Mapping[str, Any]) -> dict[str, Any]:
    error = None
    try:
        from tools.odyssey.transfer_rehearsal import rehearse

        rehearse("P1-A", str(choice.get("repo")), str(choice.get("revision")))
        return _stage(
            "RepresentationPlanner",
            FAILED,
            why=(
                "rehearse() returned without raising but this specimen is not in "
                "ARCHITECTURE_RECOGNIZER_FIXTURES; a returned plan for the wrong "
                "model is not a plan for this specimen"
            ),
            invoked=True,
        )
    except ModuleNotFoundError as exc:
        error = f"ModuleNotFoundError: {exc}"
        return _stage(
            "RepresentationPlanner",
            FAILED,
            why=(
                "invoked tools.odyssey.transfer_rehearsal.rehearse; it imports "
                "representation_library from tools/headless, which is not "
                "materialized in this sparse checkout. That is a worktree gap, "
                "not proof the producer is absent from git"
            ),
            invoked=True,
            error=error,
            evidence={
                "entry_point": "tools/odyssey/transfer_rehearsal.py:rehearse",
                "producer_in_git": bool(git("show", "HEAD:tools/headless/representation_library.py")),
                "producer_on_disk": (REPO / "tools/headless/representation_library.py").is_file(),
            },
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return _stage(
            "RepresentationPlanner",
            FAILED,
            why="rehearse() raised rather than planning this specimen",
            invoked=True,
            error=error,
        )


def stage_physical_graph_compiler(choice: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    spec_dir = Path(str(choice.get("specimen_path") or ""))
    index = spec_dir / "model.safetensors.index.json"
    collapse_present = [t for t in PGC_COLLAPSE_TENSORS if t in set(names)]
    collapse_absent = [t for t in PGC_COLLAPSE_TENSORS if t not in set(names)]
    stdout = stderr = ""
    returncode: int | None = None
    invoked = False
    if spec_dir.is_dir():
        with tempfile.TemporaryDirectory(prefix="nr-nx-generic-pgc-") as tmp:
            emit = Path(tmp) / "PHYSICAL_GRAPH_COMPILER.emit.json"
            proc = subprocess.run(
                [
                    _sys.executable,
                    str(REPO / "tools/odyssey/physical_graph_compiler.py"),
                    "--model-dir",
                    str(spec_dir),
                    "--capture",
                    str(Path(tmp) / "no-capture"),
                    "--layer",
                    "0",
                    "--emit",
                    str(emit),
                ],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=60,
            )
            invoked = True
            returncode = proc.returncode
            stdout = (proc.stdout or "")[-400:]
            stderr = (proc.stderr or "")[-1200:]
            emitted = emit.is_file()
    else:
        emitted = False
    err_line = None
    for line in (stderr or "").splitlines()[::-1]:
        if line.strip():
            err_line = line.strip()
            break
    return _stage(
        "PhysicalGraphCompiler",
        FAILED,
        why=(
            "invoked tools/odyssey/physical_graph_compiler.py --model-dir <specimen>. "
            f"returncode={returncode}. main() requires model.safetensors.index.json "
            f"(present={index.is_file()}) and an X_layer capture, and the collapse "
            "looks up MoE expert tensors this dense specimen does not have"
        ),
        invoked=invoked,
        error=err_line,
        evidence={
            "entry_point": "tools/odyssey/physical_graph_compiler.py:main",
            "returncode": returncode,
            "index_json_present": index.is_file(),
            "collapse_tensors_present": collapse_present,
            "collapse_tensors_absent": collapse_absent,
            "emit_written": emitted,
            "stderr_tail": stderr,
            "stdout_tail": stdout,
        },
    )


def stage_kernel_planner() -> dict[str, Any]:
    path = nx_audit.evidence_path(REL_KERNELS)
    if path is None:
        return _stage(
            "KernelPlanner",
            REFUSED,
            why="KERNEL_LIBRARY.json is not reachable via evidence_path; not treated as empty success",
            invoked=True,
            error="missing_kernel_library",
        )
    doc = load_json(path)
    kernels = doc.get("kernels") or []
    organs = sorted(
        {
            k.get("organ_identity")
            for k in kernels
            if isinstance(k, Mapping) and k.get("organ_identity")
        }
    )
    specimen_field = doc.get("specimen")
    return _stage(
        "KernelPlanner",
        FAILED,
        why=(
            "KERNEL_LIBRARY.json exists and was read, but it has no specimen field "
            "and catalogues qwen38 organs. The G023 stage audit already recorded "
            "treating this as AUTOMATIC-on-model-#2 as overstated. Reading the "
            "library is not a kernel plan for Qwen3-0.6B"
        ),
        invoked=True,
        evidence={
            "path": str(path),
            "n_kernels": doc.get("n_kernels") if doc.get("n_kernels") is not None else len(kernels),
            "organs": organs,
            "specimen_field": specimen_field,
            "names_this_specimen": False,
        },
    )


def stage_device_compiler(native: Mapping[str, Any]) -> dict[str, Any]:
    blocked = dict(nc.BLOCKED.get("DeviceCompiler") or {})
    return _stage(
        "DeviceCompiler",
        BLOCKED,
        why=(
            "no DeviceCompiler callable exists in tools/odyssey or hcli. "
            "noetic_compiler.BLOCKED['DeviceCompiler'] is a hardcoded note for "
            "the Qwen3-30B-A3B pipeline, not a drive of this specimen. Native "
            "GGUF match arms are "
            f"{native.get('architectures')!r}; qwen3 dense is "
            f"{'present' if _native_includes_qwen3_dense(native) else 'absent'} "
            "(qwen3moe is a different family)"
        ),
        invoked=True,
        error="no_entry_point",
        evidence={
            "noetic_compiler_blocked_why": blocked.get("why"),
            "noetic_compiler_missing_capability": blocked.get("missing_capability"),
            "copied_as_this_specimen": False,
            "native": {
                "path": native.get("path"),
                "architectures": native.get("architectures"),
                "includes_qwen2": native.get("includes_qwen2"),
                "includes_qwen3moe": native.get("includes_qwen3moe"),
                "includes_qwen3_dense": _native_includes_qwen3_dense(native),
                "includes_falcon_h1": native.get("includes_falcon_h1"),
            },
            "entry_points_searched": [
                "tools/odyssey/noetic_compiler.py (BLOCKED map, not a compiler)",
                NATIVE_LOADER,
                "hcli/agentos/flash_executable.py (Flash scaffold, Codex-owned)",
            ],
        },
    )


def stage_noetic_executable() -> dict[str, Any]:
    on_disk = (REPO / HEADLESS_FIRST_NX).is_file()
    blob = git("show", f"HEAD:{HEADLESS_FIRST_NX}") if not on_disk else (REPO / HEADLESS_FIRST_NX).read_text()
    parent_line = None
    for line in (blob or "").splitlines():
        if "qwen3.8-27b" in line.lower() or "QWEN38_PARENT" in line or "PARENT_BF16" in line:
            parent_line = line.strip()
            break
    return _stage(
        "NoeticExecutable",
        BLOCKED,
        why=(
            "the only packed-NX producer in git is tools/headless/first_noetic_executable.py, "
            "which hardlinks a Qwen3.8-27B uniform-q4 catalog and is not this specimen. "
            "It is not materialized in this sparse checkout. No generic DeviceCompiler→NX "
            "entry exists. No packed NX for Qwen3-0.6B is on disk"
        ),
        invoked=True,
        error="no_generic_packer",
        evidence={
            "producer": HEADLESS_FIRST_NX,
            "producer_on_disk": on_disk,
            "producer_in_git": bool(blob),
            "parent_line": parent_line,
            "did_not_execute_first_noetic_executable": True,
            "did_not_load_27b": True,
        },
    )


def stage_source_independence(
    choice: Mapping[str, Any],
    packed_nx: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trees = []
    if choice.get("specimen_path"):
        trees.append(str(choice["specimen_path"]))
    judged = source_independence(packed_nx, source_trees=trees)
    status = PASSED if judged["ok"] is True else FAILED
    if packed_nx is None:
        status = FAILED
    return _stage(
        "SourceIndependence",
        status,
        why=str(judged.get("why")),
        invoked=True,
        evidence=judged,
    )


def stage_dependency_accounting(packed_nx: Mapping[str, Any] | None) -> dict[str, Any]:
    needs = [
        ("serialized_nx_body", bool(packed_nx) and isinstance(nx_audit._serialized_artifact(packed_nx or {}), dict)
         and (nx_audit._serialized_artifact(packed_nx or {}) or {}).get("self_contained") is True),
        ("physical_loader", isinstance(nx_audit._loader(packed_nx or {}), dict)
         and (nx_audit._loader(packed_nx or {}) or {}).get("source_independent") is True),
        ("native_kernel_catalog", isinstance(nx_audit._kernel(packed_nx or {}), dict)),
        ("byte_ledger_closed", False),
        ("runtime_genome_digests", bool(_dot(packed_nx or {}, "reproducibility.closure_sha256"))),
    ]
    rows = [{"need": n, "present": bool(p)} for n, p in needs]
    missing = [r["need"] for r in rows if not r["present"]]
    return _stage(
        "ExecutableDependencyAccounting",
        FAILED if missing else PASSED,
        why=(
            "accounted the executable dependencies a lowered NX would have to carry; "
            f"missing={missing}"
        ),
        invoked=True,
        evidence={"dependencies": rows, "missing": missing, "packed_nx_present": packed_nx is not None},
    )


def stage_verifier(packed_nx: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(packed_nx, Mapping):
        judged = nx_audit.check_nx({})
        return _stage(
            "Verifier",
            FAILED,
            why="invoked flash_nx_audit.check_nx; there is no NX document to verify",
            invoked=True,
            evidence=judged,
        )
    judged = nx_audit.check_nx(dict(packed_nx))
    ok = judged.get("promotable") is True
    return _stage(
        "Verifier",
        PASSED if ok else FAILED,
        why=(
            "invoked tools.future.flash_nx_audit.check_nx (the landed seven-requirement "
            f"verifier); promotable={judged.get('promotable')}"
        ),
        invoked=True,
        evidence={
            "promotable": judged.get("promotable"),
            "status": judged.get("status"),
            "failed_requirements": judged.get("failed_requirements"),
            "reasons": judged.get("reasons"),
        },
    )


def first_failing_stage(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for row in stages:
        if row.get("status") != PASSED:
            return {
                "stage": row.get("stage"),
                "status": row.get("status"),
                "why": row.get("why"),
                "error": row.get("error"),
            }
    return None


def flash_nx_ready() -> dict[str, Any]:
    """Independent of the generic pipeline. False until Flash earns a packed NX."""
    loc = nx_audit.evidence_location(nx_audit.REL_NX_V0)
    if not loc.get("present"):
        return {
            "FLASH_NX_READY": False,
            "why": "FLASH_COMPLETE_V0.nx.json is not reachable; absence is not readiness",
            "path": loc.get("resolved"),
        }
    nx = load_json(loc["resolved"])
    check = nx_audit.check_nx(dict(nx))
    metadata = nx_audit._status_is_metadata_only(nx)
    ready = check.get("promotable") is True and not metadata
    return {
        "FLASH_NX_READY": False if not ready else True,
        "status": nx.get("status"),
        "promotable": check.get("promotable"),
        "metadata_only": metadata,
        "path": loc.get("resolved"),
        "failed_requirements": check.get("failed_requirements"),
        "why": (
            "Flash NX is a packed source-independent executable"
            if ready
            else f"FLASH_COMPLETE_V0.nx status={nx.get('status')!r}; check_nx.promotable={check.get('promotable')}"
        ),
    }


def emit_sleeping_lower(first: Mapping[str, Any] | None, native: Mapping[str, Any]) -> dict[str, Any]:
    wakes = [
        {
            "id": "physical_graph_compiler_accepts_dense_single_shard",
            "holds": False,
            "evidence": (
                "PhysicalGraphCompiler.main requires model.safetensors.index.json and "
                "MoE expert tensors; Qwen3-0.6B is a single-shard dense checkpoint"
            ),
        },
        {
            "id": "native_engine_match_arm_includes_qwen3_dense",
            "holds": _native_includes_qwen3_dense(native),
            "evidence": f"architectures={native.get('architectures')!r}",
        },
        {
            "id": "device_compiler_entry_point_exists",
            "holds": False,
            "evidence": "no DeviceCompiler callable; noetic_compiler.BLOCKED is a note, not a driver",
        },
        {
            "id": "generic_packer_accepts_this_specimen",
            "holds": False,
            "evidence": "first_noetic_executable.py is Qwen3.8-27B-specific",
        },
        {
            "id": "packed_source_independent_nx_on_disk",
            "holds": False,
            "evidence": "no NX body for Qwen3-0.6B; FLASH_COMPLETE_V0.nx is a metadata seal of a different model",
        },
    ]
    holding = [w["id"] for w in wakes if not w["holds"]]
    reason = (
        f"NX_LOWER blocked at stage {(first or {}).get('stage')}: {(first or {}).get('error') or (first or {}).get('why')}"
    )
    unit = wus.emit_hcli_workunit(
        id="future.nr-nx-generic.sleep.nx-lower",
        role="science",
        description=f"SLEEPING until a generic NR→NX packer accepts the chosen specimen. {reason}",
        dependencies=[],
        resource_class="COMPILE",
        verifier="future.nr_nx_generic.source_independence",
        provider="future.nr_nx_generic",
        effect_class="READ_ONLY",
        preferred_backend="cpu",
        status="sleeping",
        classification="SLEEPING",
        extras={
            "sleeping": True,
            "blocked_reason": reason,
            "requires_quiescence": False,
            "synthetic_result_forbidden": True,
            "wake_unmet": holding,
            "first_failing_stage": (first or {}).get("stage"),
        },
    )
    wus.validate_emitted_unit(unit)
    if unit.get("status") in {"pending", "PENDING", "ready", "READY"}:
        raise ValueError(f"sleeping unit leaked status={unit.get('status')!r}")
    return {
        "id": unit["id"],
        "status": unit["status"],
        "classification": unit.get("classification"),
        "resource_class": unit.get("resource_class"),
        "verifier": unit.get("verifier"),
        "blocked_reason": unit.get("blocked_reason"),
        "wake_unmet": holding,
        "wake_conditions": wakes,
        "synthetic_result_forbidden": True,
        "first_failing_stage": (first or {}).get("stage"),
        "exact_error": (first or {}).get("error"),
    }


# ---------------------------------------------------------------------------
# Assemble.
# ---------------------------------------------------------------------------


def _load_specimen_inputs(choice: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str], str]:
    if choice.get("ok") is not True or not choice.get("specimen_path"):
        return None, [], "absent"
    spec_dir = Path(str(choice["specimen_path"]))
    cfg_path = spec_dir / "config.json"
    if not cfg_path.is_file():
        return None, [], "missing_config"
    cfg = json.loads(cfg_path.read_text())
    names, via = _tensor_names(spec_dir)
    return cfg, names, via


def assemble() -> dict[str, Any]:
    choice = choose_specimen()
    cfg, names, names_via = _load_specimen_inputs(choice)
    native = native_engine_architectures()

    stages: list[dict[str, Any]] = []
    stages.append(stage_specimen_select(choice))
    stages.append(stage_specimen_present(choice))
    arch = stage_architecture_recognizer(choice, cfg=cfg, names=names, names_via=names_via)
    stages.append(arch)
    og = stage_organ_graph(arch, cfg=cfg, names=names)
    stages.append(og)
    stages.append(stage_nr_identify(choice, arch, og))
    stages.append(stage_doctor(choice, names))
    stages.append(stage_representation_planner(choice) if choice.get("ok") else _stage(
        "RepresentationPlanner", REFUSED, why="no specimen selected", invoked=False
    ))
    stages.append(
        stage_physical_graph_compiler(choice, names)
        if choice.get("ok") and Path(str(choice.get("specimen_path") or "")).is_dir()
        else _stage("PhysicalGraphCompiler", REFUSED, why="specimen directory not on disk", invoked=False)
    )
    stages.append(stage_kernel_planner())
    stages.append(stage_device_compiler(native))
    stages.append(stage_noetic_executable())

    packed_nx = None
    packed_path = None
    stages.append(stage_source_independence(choice, packed_nx))
    stages.append(stage_dependency_accounting(packed_nx))
    stages.append(stage_verifier(packed_nx))

    if [s["stage"] for s in stages] != list(STAGE_ORDER):
        raise StageSkipForbidden(
            f"stage order drifted: {[s['stage'] for s in stages]} != {list(STAGE_ORDER)}"
        )
    for row in stages:
        if row["status"] in FORBIDDEN_STAGE_STATUS:
            raise StageSkipForbidden(f"{row['stage']} leaked {row['status']}")

    try:
        callable_ok = declare_pipeline_callable(stages, packed_nx_path=packed_path)
    except PipelineCallableForbidden:
        callable_ok = False
    first = first_failing_stage(stages)
    nx_lower_names = {
        "PhysicalGraphCompiler",
        "DeviceCompiler",
        "NoeticExecutable",
    }
    first_nx_lower = next(
        (
            {
                "stage": row["stage"],
                "status": row["status"],
                "why": row["why"],
                "error": row.get("error"),
            }
            for row in stages
            if row["stage"] in nx_lower_names and row["status"] != PASSED
        ),
        None,
    )
    flash = flash_nx_ready()
    sleeping = emit_sleeping_lower(first_nx_lower or first, native)

    launch_still_flash = (
        "odyssey_launch._eval_nr_nx still keys nr_nx_path_callable on "
        "FLASH_NX_COMPLETENESS_AUDIT.seven_all_met and FLASH_COMPLETE_V0.nx status. "
        "That criterion is Flash-specific. This lane does not rewrite it. The "
        "bootstrap architecture (Qwen27 launches Odyssey; Flash is an evolving "
        "child) says the generic path should be enough; the generic path is not "
        "callable today, so the Flash-specific gate stays standing for two "
        "independent reasons, not one collapsed reason"
    )

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Drive the real NR→NX compiler stages on the cheapest whole-tree-verified "
            "specimen and keep GENERIC_NR_NX_PIPELINE_CALLABLE separate from FLASH_NX_READY"
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "is_a_measurement": False,
        "GENERIC_NR_NX_PIPELINE_CALLABLE": False if not callable_ok else True,
        "FLASH_NX_READY": flash["FLASH_NX_READY"],
        "facts_are_independent": True,
        "specimen": choice,
        "native_engine": {
            "path": native.get("path"),
            "architectures": native.get("architectures"),
            "includes_qwen2": native.get("includes_qwen2"),
            "includes_qwen3moe": native.get("includes_qwen3moe"),
            "includes_qwen3_dense": _native_includes_qwen3_dense(native),
            "includes_falcon_h1": native.get("includes_falcon_h1"),
            "ok": native.get("ok"),
        },
        "stages": stages,
        "first_failing_stage": first,
        "first_nx_lower_failure": first_nx_lower,
        "flash": flash,
        "launch_criterion_still_flash_specific": launch_still_flash,
        "sleeping_workunit": sleeping,
        "physical_ebpw": None,
        "physical_ebpw_written": False,
        "compiler_entry_points": {
            "ArchitectureRecognizer": "tools/odyssey/arch_recognizer.py:recognize",
            "OrganGraph": "tools/odyssey/physical_graph_compiler.py:organ_graph",
            "Doctor": "tools/odyssey/doctor_tournament.py (PARENT hardcoded to Qwen3.8-27B)",
            "RepresentationPlanner": "tools/odyssey/transfer_rehearsal.py:rehearse",
            "PhysicalGraphCompiler": "tools/odyssey/physical_graph_compiler.py:main",
            "KernelPlanner": "receipts/headless/KERNEL_LIBRARY.json (no specimen field)",
            "DeviceCompiler": "NO CALLABLE; tools/odyssey/noetic_compiler.py BLOCKED map",
            "NoeticExecutable": "tools/headless/first_noetic_executable.py (Qwen38 27B only)",
            "native_loader": NATIVE_LOADER,
            "nx_verifier": "tools/future/flash_nx_audit.py:check_nx",
        },
        "recovered_implementation": [
            "tools/future/nr_nx_path.py — seven-requirement map, SLEEPING units, physical_ebpw refusal; EXTENDED, not forked",
            "tools/future/flash_nx_audit.py — check_nx, evidence_path, METADATA_ONLY, synthetic_promotable_nx",
            "tools/future/flash_nr_complete.py — composition NR is not serialized_nr_information",
            "tools/future/ebpw_categories.py — typed EBPW; physical remains unwritten",
            "tools/future/specimen_verify.py — WHOLE_TREE_VERIFIED list; ModelLake not mutated",
            "tools/odyssey/arch_recognizer.py — invoked on local config+names, no network, no weights",
            "tools/odyssey/physical_graph_compiler.py — organ_graph invoked; main() subprocessed and failed",
            "tools/odyssey/doctor_tournament.py — PARENT/PROBE_TENSORS read, probes() not called",
            "tools/odyssey/transfer_rehearsal.py — rehearse() invoked, ModuleNotFoundError recorded",
            "tools/odyssey/noetic_compiler.py — BLOCKED map cited as a note, not as a drive of 0.6B",
            "crates/hawking-core/src/model/mod.rs — shipping GGUF match arms",
            "tools/future/odyssey_launch.py _eval_nr_nx — Flash-specific launch criterion, read not rewritten",
            "tools/future/workunit_species.py emit_hcli_workunit — SLEEPING unit, never pending",
        ],
        "gaps_closed": [
            "generic NR→NX stages driven on a real specimen instead of inferred from Flash's missing NX",
            "GENERIC_NR_NX_PIPELINE_CALLABLE and FLASH_NX_READY recorded as separate facts",
            "source-independence checker that fails a runtime read into the source tree and a renamed source pointer",
            "PhysicalGraphCompiler live invocation captured: FileNotFoundError on model.safetensors.index.json",
            "pipeline_callable refuses SKIPPED stages and refuses a pass without a packed NX",
        ],
        "negative_findings": [
            "GENERIC_NR_NX_PIPELINE_CALLABLE is False on Qwen3-0.6B",
            "FLASH_NX_READY is False; FLASH_COMPLETE_V0.nx remains SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
            "PhysicalGraphCompiler cannot run on a dense single-shard specimen",
            "Doctor is hardcoded to Qwen3.8-27B tensor names",
            "RepresentationPlanner cannot import tools/headless/representation_library in this checkout",
            "native engine match arms include qwen2 and qwen3moe, not dense qwen3, not falcon_h1",
            "no generic NX packer; first_noetic_executable is a 27B mix",
            "no physical EBPW was written",
        ],
        "what_this_cannot_establish": [
            "a packed source-independent NX for Qwen3-0.6B, Falcon-H1, or Flash",
            "that parameterizing PhysicalGraphCompiler for dense MLP would produce a correct collapse",
            "that adding a qwen3 GGUF match arm would load this safetensors specimen",
            "protected complete-token performance or physical EBPW",
            "that Odyssey I can launch; the Flash-specific criterion still stands",
        ],
        "next_workunits": [
            {
                "id": "WU.CPU.nr-nx-generic.parameterize-physical-graph-compiler",
                "schedule": "CPU_NEXT",
                "owner": "Codex (tools/odyssey/physical_graph_compiler.py is Codex-owned)",
                "wake": "accept dense single-shard specimens and dense mlp.gate_proj paths",
            },
            {
                "id": sleeping["id"],
                "schedule": SLEEPING,
                "wake_unmet": sleeping["wake_unmet"],
            },
        ],
        "resident_callable": {
            "entry_point": "tools.future.nr_nx_generic.build()",
            "workunit": (
                "one CPU_ANALYSIS unit; drive compiler stages on a real specimen; "
                "no GPU authority; no packer"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_EXECUTION.complete-token",
            "fails_closed": (
                "absent specimen/config is REFUSED; a stage that cannot run is FAILED/"
                "BLOCKED by name, never SKIPPED; source independence fails on a source-tree "
                "read; GENERIC_NR_NX_PIPELINE_CALLABLE cannot be True if any stage was "
                "skipped or no packed NX exists; physical_ebpw cannot be written"
            ),
        },
    }
    assert_no_physical_ebpw(doc)
    if doc["GENERIC_NR_NX_PIPELINE_CALLABLE"] is True and packed_path is None:
        raise PipelineCallableForbidden("callable True with no packed NX")
    if doc["FLASH_NX_READY"] is True and flash.get("metadata_only"):
        raise PipelineCallableForbidden("FLASH_NX_READY True on a metadata seal")
    return doc


def build() -> Path:
    doc = assemble()
    assert_no_physical_ebpw(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
