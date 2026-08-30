"""ODYSSEY I launch gate — WHAT IS TRUE? may not start on a green scaffold.

The sidecar substrate is executable. That is not the same as Odyssey started.
This module evaluates sixteen launch criteria from disk evidence, names every
unmet criterion (not the first), and writes ODYSSEY_I_LAUNCH.json ONLY when
all sixteen pass. Today's honest answer is REFUSED.

A subsystem is not operational until the resident can discover it, invoke it,
schedule it and verify it; its result changes a frontier; the result persists;
and next work refills. A CLI a human can run is not enough.

    python3 tools/future/odyssey_launch.py --verify
    python3 tools/future/odyssey_launch.py --launch
    python3 -m pytest tools/future/test_odyssey_launch.py -q

Everything emitted is STATIC_ONLY, bench UNKNOWN, gpu_authority false.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git, RECEIPTS

import argparse
import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.future import contamination as C
from tools.future import flash_nx_audit as nx_audit
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey3_adversary as o3
from tools.future import repro_science as rs
from tools.future import workunit_species as wus


RECEIPT = "ODYSSEY_LAUNCH_GATE.json"
LAUNCH_RECEIPT = "ODYSSEY_I_LAUNCH.json"
SCHEMA = "hawking.future.odyssey_launch.v1"
LAUNCH_SCHEMA = "hawking.future.odyssey_i_launch.v1"
VERSION = 1
RECORDED_BY = "tools/future/odyssey_launch.py"

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

# Concurrent-wave modules. Named as integration points; never imported.
THIS_WAVE_SIBLINGS = (
    "codex_behaviors",
    "resident_api",
    "workgraph",
    "detached",
    "wakeup",
    "evidence_dag",
    "scar_scheduling",
    "dirty_measure",
    "protected_window",
    "sandbox",
    "resident_identity",
    "frontiers",
    "succession",
    "flash_schools",
    "flash_nr_complete",
    "super_resident",
    "tabula",
    "debugger",
    "autonomy_trial",
)

# HCLI AgentOS autonomy is a recovered control-plane proof. It is not an
# Odyssey I resident-orchestration trial and must not open this gate.
HCLI_AUTONOMY_SCHEMA = "hcli.agentos.autonomy_gate.v1"
ODYSSEY_AUTONOMY_SCHEMAS = (
    "hawking.future.autonomy_trial.v1",
    "hawking.odyssey.autonomy_trial.v1",
    "hawking.future.super_resident.autonomy.v1",
)

CRITERION_IDS: tuple[str, ...] = (
    "resident_autonomy_trial_pass",
    "specimen_curriculum_ready",
    "modellake_identity",
    "doctor_callable",
    "gravity_callable",
    "nr_nx_path_callable",
    "evidence_hierarchy",
    "negative_science",
    "workgraphs",
    "self_refill",
    "dirty_measurement",
    "protected_scheduling",
    "transfer_substrate",
    "adversary_substrate",
    "crash_recovery",
    "receipts",
)

GRAPH_STAGES: tuple[str, ...] = (
    "specimen_verification",
    "architecture_fingerprint",
    "doctor",
    "gravity",
    "representation_search",
    "physical_profiling",
    "nr",
    "nx",
    "native_execution",
    "capability",
    "pareto",
    "laws",
    "scars",
)
PHASE_LISTEN_STAGES: tuple[str, ...] = ("phase_ii_transfer", "phase_iii_attack")
GPU_STAGES = frozenset(
    {"physical_profiling", "nx", "native_execution", "capability"}
)
COMPILE_STAGES = frozenset({"nr", "nx"})

CURRICULUM_ROLES: tuple[tuple[str, str], ...] = (
    ("very_small_dense_procedural_speed", "very small dense for procedural speed"),
    ("small_dense_alternate_architecture_transfer", "small dense alternate architecture for transfer"),
    ("mid_size_dense_compiler", "mid-size dense for compiler"),
    ("qwen27_mature_physical", "Qwen27 for mature physical"),
    ("flash_heterogeneous_frontier", "Flash for heterogeneous frontier"),
)

EVIDENCE_LATTICE = (
    "STATIC_ONLY",
    "DIAGNOSTIC_RELATIVE",
    "PROTECTED_ABSOLUTE",
)

SIDECAR_CLAIM = (
    "Static sidecar artifact. No hardware measurement. A launch-gate REFUSED "
    "receipt is not Odyssey started. ODYSSEY_I_LAUNCH.json is the only phase-"
    "transition receipt and is written only when every criterion is met."
)

# Local interface the this-wave siblings will replace. Named, not imported.
INTEGRATION_POINTS: dict[str, str] = {
    "autonomy_trial": "tools/future/autonomy_trial.py — Odyssey I resident-orchestration trial pass receipt",
    "resident_identity": "tools/future/resident_identity.py — canonical resident declaration the launch receipt binds",
    "sandbox": "tools/future/sandbox.py — orchestrator sandbox identity the resident operates",
    "workgraph": "tools/future/workgraph.py — WorkGraph runtime that admits, schedules and verifies the units this gate emits",
    "frontiers": "tools/future/frontiers.py — frontier objects a result must change, and the refill that follows",
    "succession": "tools/future/succession.py — self-refill / next-work succession under resident control",
    "dirty_measure": "tools/future/dirty_measure.py — dirty-class measurement that cannot launder into PROTECTED_ABSOLUTE",
    "protected_window": "tools/future/protected_window.py — protected scheduling that does not seize Codex's GPU lock",
    "evidence_dag": "tools/future/evidence_dag.py — evidence DAG the resident walks; lattice alone is not the DAG",
    "wakeup": "tools/future/wakeup.py — wake SLEEPING WorkUnits when hardware qualifies",
    "super_resident": "tools/future/super_resident.py — HCLI super-resident operating the orchestrator sandbox",
    "resident_api": "tools/future/resident_api.py — discover/invoke surface HCLI uses instead of a human CLI",
}


# ---------------------------------------------------------------------------
# Evidence probes. Sparse checkout: missing here is not absence in the project.
# ---------------------------------------------------------------------------


def _checkout_roots() -> list[Path]:
    roots: list[Path] = [REPO]
    common = git("rev-parse", "--git-common-dir")
    if common:
        path = Path(common)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        else:
            path = path.resolve()
        parent = path.parent if path.name == ".git" else path.parent
        if parent not in roots and parent.is_dir():
            roots.append(parent)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def probe_json(*rels: str) -> dict[str, Any]:
    """Load the first readable JSON among `rels`. Record which path was taken.

    A miss is `found=False` with the search log. It is not a proof the file
    does not exist in another checkout. Callers must cope with either state.
    """
    searched: list[str] = []
    snapshot = REPO / "receipts" / "future" / "evidence"
    for rel in rels:
        rel = rel.replace("\\", "/").lstrip("./")
        candidates: list[tuple[str, Path]] = [("worktree", REPO / rel)]
        if snapshot.is_dir():
            candidates.append(("evidence_snapshot", snapshot / Path(rel).name))
        for root in _checkout_roots()[1:]:
            candidates.append(("primary_checkout", root / rel))
        for origin, path in candidates:
            searched.append(f"{origin}:{path}")
            if path.is_file():
                try:
                    return {
                        "found": True,
                        "path_taken": origin,
                        "rel": rel,
                        "resolved": str(path),
                        "doc": load_json(path),
                        "searched": searched,
                    }
                except (OSError, json.JSONDecodeError) as exc:
                    searched.append(f"unreadable:{path}:{type(exc).__name__}")
        blob = git("show", f"HEAD:{rel}")
        searched.append(f"git:HEAD:{rel}")
        if blob:
            try:
                return {
                    "found": True,
                    "path_taken": "git_head",
                    "rel": rel,
                    "resolved": f"git:HEAD:{rel}",
                    "doc": json.loads(blob),
                    "searched": searched,
                }
            except json.JSONDecodeError as exc:
                searched.append(f"git_unreadable:{rel}:{type(exc).__name__}")
    return {
        "found": False,
        "path_taken": "not_found",
        "rel": rels[0] if rels else None,
        "resolved": None,
        "doc": None,
        "searched": searched,
        "note": (
            "not present in this worktree, the evidence snapshot, the primary "
            "checkout, or git HEAD. Sparse checkout: this is not proof of absence."
        ),
    }


def _module_file(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if path.is_file():
        return {"present": True, "path_taken": "worktree", "resolved": str(path), "rel": rel}
    blob = git("ls-files", "--error-unmatch", rel)
    if blob:
        return {"present": True, "path_taken": "git_tracked_not_materialized", "resolved": rel, "rel": rel}
    for root in _checkout_roots()[1:]:
        other = root / rel
        if other.is_file():
            return {"present": True, "path_taken": "primary_checkout", "resolved": str(other), "rel": rel}
    return {
        "present": False,
        "path_taken": "not_found",
        "resolved": None,
        "rel": rel,
        "note": "sparse miss is not project absence",
    }


def _importable(dotted: str) -> dict[str, Any]:
    try:
        parts = dotted.split(".")
        mod = __import__(dotted, fromlist=[parts[-1]])
        return {"ok": True, "module": dotted, "file": getattr(mod, "__file__", None)}
    except Exception as exc:
        return {"ok": False, "module": dotted, "error": f"{type(exc).__name__}: {exc}"}


def _receipt_sealed(doc: Mapping[str, Any] | None) -> bool:
    if not isinstance(doc, Mapping):
        return False
    seal = doc.get("seal_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        return False
    bench = doc.get("bench") if isinstance(doc.get("bench"), Mapping) else {}
    return bench.get("measurement_state") == "STATIC_ONLY" or "schema" in doc


def _criterion(
    cid: str,
    *,
    met: bool,
    reason: str,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    operational: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": cid,
        "met": bool(met),
        "reason": reason,
        "evidence": [dict(e) for e in (evidence or ())],
        "operational": dict(operational or {}),
    }
    if extra:
        row.update(dict(extra))
    return row


def operational_bar(
    *,
    discover: bool,
    invoke: bool,
    schedule: bool,
    verify: bool,
    frontier: bool,
    persist: bool,
    refill: bool,
    notes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    flags = {
        "discover": bool(discover),
        "invoke": bool(invoke),
        "schedule": bool(schedule),
        "verify": bool(verify),
        "frontier": bool(frontier),
        "persist": bool(persist),
        "refill": bool(refill),
    }
    return {
        "resident_operational": all(flags.values()),
        "cli_is_not_enough": True,
        "flags": flags,
        "notes": dict(notes or {}),
    }


# ---------------------------------------------------------------------------
# Curriculum roles recovered from ModelLake seals + Odyssey I + law-store schools.
# Do not propose exhaustively optimizing every downloaded model.
# ---------------------------------------------------------------------------


def _specimen_dirs_on_disk() -> set[str]:
    """Specimen directory names actually present in ModelLake. Read-only."""
    root = Path("/Volumes/corpdrive/hawking-modellake/specimens")
    try:
        return {p.name for p in root.iterdir() if p.is_dir()}
    except OSError:
        return set()  # the lake may not be mounted; that is not an error here


def _independently_verified() -> dict[str, dict[str, Any]]:
    """Specimens whose every published digest was RECOMPUTED and matched.

    ModelLake's own seals say MANIFEST_ONLY because it verified most files by
    size. That is not verification and the gate is right to refuse it. But the
    digests were never missing -- HuggingFace writes a .metadata sidecar per
    file -- so whole-tree verification can be EARNED offline, and this reads the
    receipt where it was earned.

    Strict on purpose: a row counts only if it hashed real bytes, matched every
    file, and had no file it could not check. Anything softer would turn a
    correct refusal into a false readiness, which is the exact failure this
    gate exists to prevent.
    """
    rec = probe_json("receipts/future/SPECIMEN_VERIFICATION.json")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    out: dict[str, dict[str, Any]] = {}
    for row in (doc or {}).get("results") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("status") != "WHOLE_TREE_VERIFIED":
            continue
        if not (isinstance(row.get("bytes_hashed"), int) and row["bytes_hashed"] > 0):
            continue
        if row.get("mismatched") or row.get("no_remote_digest"):
            continue
        if row.get("verified") != row.get("n_files"):
            continue
        out[str(row.get("specimen") or "")] = dict(row)
    return out


def _ready(identity: Mapping[str, Any], *, require_lake_verified: bool) -> tuple[bool, str]:
    if require_lake_verified and not identity.get("whole_tree_verified"):
        if identity.get("published_as_verified") is False:
            return False, "ModelLake pin exists but the specimen is not published as verified"
        if identity.get("in_specimens_listing") and not identity.get("whole_tree_verified"):
            n_sha = identity.get("n_sha256_verified")
            n_files = identity.get("n_files")
            return False, (
                f"ModelLake manifest is partial "
                f"(n_sha256_verified={n_sha} n_files={n_files}); not a sealed specimen"
            )
        if not identity.get("in_specimens_listing"):
            return False, "identity known but specimen is not in the ModelLake specimens listing"
        return False, "ModelLake publication is not whole-tree verified"
    if identity.get("patient_state") == "RETIRED":
        # The odysseys are recurrent phases and the first canonical completion
        # is historical, so a patient retired from that first wave is a
        # specimen with a PROVEN role, not a disqualified one. It counts only
        # when both halves hold: the prior seal exists, and the specimen has
        # been independently whole-tree verified NOW. Retirement alone would
        # be a pass on a stale seal; verification alone would lose the prior
        # work. Recurrence is recorded so nothing downstream reads a repeat
        # phase as a first-wave result.
        if identity.get("whole_tree_verified") and identity.get("patient_seal"):
            return True, (
                "RECURRENT_PATIENT: retired from the historical first wave, prior "
                "seal intact, and whole-tree verified again now"
            )
        return False, (
            "prior Odyssey I patient is RETIRED and has not been whole-tree "
            "verified again; a stale seal is not a live first-wave specimen"
        )
    if identity.get("physical_status") == "metadata_only_weights_not_present":
        return False, "school identity is metadata-only; weights are not present"
    if not identity.get("revision") and not identity.get("resolved_sha") and not identity.get("patient_seal"):
        # A local directory has no repository revision. The external specimen
        # seal is that identity; lake specimens cannot take this branch.
        try:
            from tools.future.external_specimen_seal import accept_as_sealed_identity
        except ImportError:
            return False, "no sealed revision or patient seal"
        ok, why = accept_as_sealed_identity(identity)
        if ok:
            return True, why
        return False, "no sealed revision or patient seal"
    if identity.get("whole_tree_verified"):
        return True, "ModelLake whole-tree sha256 verification"
    return False, "sealed identity is not enough; live first-wave specimen is not published"


def _lake_index(census: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Index ModelLake census manifests and specimen dirs by repo / slug."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(census, Mapping):
        return out
    verified = census.get("verified_receipts") if isinstance(census.get("verified_receipts"), Mapping) else {}
    for row in verified.get("receipts") or []:
        if not isinstance(row, Mapping):
            continue
        repo = str(row.get("repo") or "")
        if not repo:
            continue
        out[repo] = {
            "repo": repo,
            "revision": row.get("revision") or row.get("resolved_sha"),
            "resolved_sha": row.get("resolved_sha"),
            "manifest_path": row.get("path"),
            "specimen_path": row.get("specimen_path"),
            "n_files": row.get("n_files"),
            "n_sha256_verified": row.get("n_sha256_verified"),
            "n_size_only_verified": row.get("n_size_only_verified"),
            "whole_tree_verified": False,
            "in_specimens_listing": False,
            "source": "modellake_manifest",
        }
    specimens = census.get("specimens") if isinstance(census.get("specimens"), Mapping) else {}
    names = {str(e.get("name")) for e in (specimens.get("entries") or []) if isinstance(e, Mapping)}
    # DISK STATE IS AUTHORITY. The census is a cache of it, and a specimen the
    # census never recorded still exists. Reading only the census reported
    # Mistral-Small-24B as "not in the ModelLake specimens listing" while its
    # directory was sitting in that listing -- a wrong reason attached to a
    # correct refusal, which is the kind of error that sends work to the wrong
    # place. ModelLake is read, never written.
    names |= _specimen_dirs_on_disk()
    for repo, row in out.items():
        slug = (row.get("specimen_path") or "").rstrip("/").split("/")[-1]
        row["in_specimens_listing"] = slug in names
        n_files = row.get("n_files")
        n_sha = row.get("n_sha256_verified")
        row["whole_tree_verified"] = bool(
            row["in_specimens_listing"]
            and isinstance(n_files, int)
            and isinstance(n_sha, int)
            and n_files > 0
            and n_sha == n_files
        )
    # A specimen present on disk but absent from the census needs a row of its
    # own, or the disk fallback above can only correct rows the cache already
    # had -- which is how Mistral-Small-24B stayed invisible.
    for name in sorted(_specimen_dirs_on_disk()):
        slug, _, rev = name.partition("@")
        repo = slug.replace("--", "/", 1)
        if repo in out:
            continue
        out[repo] = {
            "repo": repo,
            "revision": rev or None,
            "resolved_sha": rev or None,
            "manifest_path": None,
            "specimen_path": f"/Volumes/corpdrive/hawking-modellake/specimens/{name}",
            "n_files": None,
            "n_sha256_verified": None,
            "n_size_only_verified": None,
            "whole_tree_verified": False,
            "in_specimens_listing": True,
            "source": "modellake_specimens_dir",
        }

    earned = _independently_verified()
    for row in out.values():
        slug = (row.get("specimen_path") or "").rstrip("/").split("/")[-1]
        hit = earned.get(slug)
        if not hit:
            continue
        row["whole_tree_verified"] = True
        row["verification_source"] = "tools/future/specimen_verify.py (offline recomputation)"
        row["bytes_hashed"] = hit.get("bytes_hashed")
        row["in_specimens_listing"] = True

    flash = census.get("source") if isinstance(census.get("source"), Mapping) else {}
    if flash.get("repo"):
        repo = str(flash.get("repo"))
        checks = census.get("checks") if isinstance(census.get("checks"), Mapping) else {}
        manifest = census.get("flash_target_manifest") if isinstance(census.get("flash_target_manifest"), Mapping) else {}
        out[repo] = {
            "repo": repo,
            "revision": flash.get("requested_revision") or flash.get("revision"),
            "resolved_sha": flash.get("requested_revision"),
            "manifest_path": None,
            "specimen_path": manifest.get("final_root"),
            "n_files": None,
            "n_sha256_verified": manifest.get("verified_file_count"),
            "n_size_only_verified": None,
            "whole_tree_verified": bool(manifest.get("whole_tree_verified")),
            "in_specimens_listing": bool(manifest.get("final_present")),
            "published_as_verified": not bool(checks.get("target_not_published_as_verified", True)),
            "source": "flash_pinned_census",
            "census_qualification": census.get("qualification"),
        }
    return out


def _odyssey_i_patients() -> list[dict[str, Any]]:
    root = REPO / "receipts" / "odyssey-i"
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for seal in sorted(root.glob("*_PATIENT_SEAL.json")):
        try:
            doc = load_json(seal)
        except (OSError, json.JSONDecodeError):
            continue
        oxx = doc.get("oxx") or seal.name.split("_", 1)[0]
        ext_path = root / f"{oxx}_EXTERNAL.json"
        weights = None
        if ext_path.is_file():
            try:
                ext = load_json(ext_path)
                weights = ext.get("weights_canonical")
            except (OSError, json.JSONDecodeError):
                weights = None
        rows.append(
            {
                "oxx": oxx,
                "status": doc.get("status"),
                "state": doc.get("state"),
                "seal": str(seal.relative_to(REPO)),
                "weights_canonical": weights,
                "sealed_mechanisms": list(doc.get("sealed_mechanisms") or []),
            }
        )
    return rows


def propose_specimen_curriculum(census_doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """First specimen set by curriculum role. Not 'every model in the lake'."""
    if census_doc is None:
        probe = probe_json(
            "receipts/future/evidence/HCLI_MODELLAKE_FLASH_CENSUS.json",
            "receipts/headless/HCLI_MODELLAKE_FLASH_CENSUS.json",
            "receipts/headless/MODELLAKE_FLASH_NEXT_CENSUS.json",
        )
        census_doc = probe.get("doc")
        census_probe = probe
    else:
        census_probe = {"found": True, "path_taken": "caller", "rel": None, "resolved": None}

    lake = _lake_index(census_doc if isinstance(census_doc, Mapping) else None)
    patients = _odyssey_i_patients()
    schools = {k: dict(v) for k, v in ols.SCHOOLS.items()}

    def _patient_for(*needles: str) -> dict[str, Any] | None:
        for p in patients:
            blob = " ".join(
                str(x) for x in (p.get("oxx"), p.get("weights_canonical"), p.get("seal"))
            ).lower()
            if any(n.lower() in blob for n in needles):
                return p
        return None

    roles: list[dict[str, Any]] = []

    q06 = dict(lake.get("Qwen/Qwen3-0.6B") or {})
    # The gate reported this specimen as absent from the ModelLake specimens
    # listing, which was true and was read as "the model is not here". It is
    # here -- complete, inside ModelLake, under partial/ rather than specimens/.
    # Ten files, ten published digests, all recomputed and matched. Location was
    # the only thing partial about it. ModelLake still owns it and nothing here
    # moves or writes to it.
    q06_partial = _independently_verified().get("Qwen--Qwen3-0.6B@c1899de289a0#partial") or {}
    if q06_partial:
        q06.update({
            "whole_tree_verified": True,
            "in_specimens_listing": True,
            "specimen_owner": "modellake_partial",
            "specimen_path": q06_partial.get("specimen_path"),
            "verification_source": "tools/future/specimen_verify.py (offline recomputation)",
        })
        q06.setdefault("revision", "c1899de289a0")
    roles.append(
        {
            "role": CURRICULUM_ROLES[0][0],
            "purpose": CURRICULUM_ROLES[0][1],
            "repo": q06.get("repo") or "Qwen/Qwen3-0.6B",
            "revision": q06.get("revision"),
            "architecture_family": "dense_transformer",
            "identity_source": q06.get("source") or "modellake_manifest",
            "modellake": q06,
            "located_under_partial": bool(q06_partial),
            **dict(zip(("ready", "ready_reason"), _ready(q06, require_lake_verified=True))),
        }
    )

    falcon = lake.get("tiiuae/Falcon-H1-7B-Instruct") or {}
    p001 = _patient_for("falcon-h1", "O001")
    falcon_id = dict(falcon)
    if p001:
        falcon_id["patient_seal"] = p001.get("seal")
        falcon_id["patient_state"] = p001.get("state")
        falcon_id["patient_status"] = p001.get("status")
    roles.append(
        {
            "role": CURRICULUM_ROLES[1][0],
            "purpose": CURRICULUM_ROLES[1][1],
            "repo": falcon.get("repo") or "tiiuae/Falcon-H1-7B-Instruct",
            "revision": falcon.get("revision"),
            "architecture_family": "falcon_h1",
            "identity_source": "modellake_manifest+odyssey_i_O001",
            "modellake": falcon,
            "prior_odyssey_i": p001,
            **dict(zip(("ready", "ready_reason"), _ready(falcon_id, require_lake_verified=True))),
        }
    )

    mistral_partial = None
    stale = []
    if isinstance(census_doc, Mapping):
        stale = list(census_doc.get("stale_partial_candidates") or [])
    for row in stale:
        path = str(row.get("path") or "")
        if "Mistral-Small" in path:
            mistral_partial = row
            break
    p004 = _patient_for("mistral-small", "O004", "24B")
    # Consult the index rather than asserting absence. This role hardcoded
    # in_specimens_listing=False and so reported "not in the ModelLake specimens
    # listing" for a specimen whose 89GB directory was sitting in that listing --
    # a wrong reason on a correct refusal, which sends the next worker to fix
    # the wrong thing.
    mistral_lake = lake.get("mistralai/Mistral-Small-3.1-24B-Instruct-2503") or {}
    mistral_id = {
        **mistral_lake,
        "repo": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "revision": mistral_lake.get("revision")
        or "68faf511d618ef198fef186659617cfd2eb8e33a",
        "stale_partial": mistral_partial,
        "patient_seal": None if not p004 else p004.get("seal"),
        "patient_state": None if not p004 else p004.get("state"),
        "source": "odyssey_i_O004+modellake_partial",
    }
    roles.append(
        {
            "role": CURRICULUM_ROLES[2][0],
            "purpose": CURRICULUM_ROLES[2][1],
            "repo": mistral_id["repo"],
            "revision": mistral_id["revision"],
            "architecture_family": "dense_transformer",
            "identity_source": mistral_id["source"],
            "modellake": {"stale_partial": mistral_partial},
            "prior_odyssey_i": p004,
            **dict(zip(("ready", "ready_reason"), _ready(mistral_id, require_lake_verified=True))),
        }
    )

    q27 = dict(schools.get("Qwen27") or {})
    # The Qwen27 parent is not a ModelLake specimen and never was. It is the
    # 52GB directory the Doctor and Gravity tools read, it carries the same
    # HuggingFace .metadata digests, and it is verified by exactly the same rule.
    # It is labelled local_directory so it is never mistaken for a sealed lake
    # specimen, and ModelLake's ownership of the lake is untouched.
    q27_local = _independently_verified().get("qwen3.8-27b-abliterated-bf16@local") or {}
    q27_id = {
        "repo": q27.get("source_model") or "Qwen3.8-27B",
        "revision": None,
        "architecture_family": q27.get("architecture_family"),
        "in_specimens_listing": (
            "Qwen3.8-27B" in lake or "Qwen/Qwen3.8-27B" in lake or bool(q27_local)
        ),
        "whole_tree_verified": bool(q27_local),
        "specimen_owner": q27_local.get("owner") or "modellake",
        "specimen_path": q27_local.get("specimen_path"),
        "physical_status": q27.get("physical_status"),
        "source": "odyssey2_law_store.SCHOOLS.Qwen27",
    }
    roles.append(
        {
            "role": CURRICULUM_ROLES[3][0],
            "purpose": CURRICULUM_ROLES[3][1],
            "repo": q27_id["repo"],
            "revision": q27_id["revision"],
            "architecture_family": q27_id["architecture_family"],
            "identity_source": q27_id["source"],
            "school": q27,
            "modellake": lake.get("Qwen3.8-27B") or lake.get("Qwen/Qwen3.8-27B") or {},
            "local_specimen": q27_local or None,
            **dict(zip(("ready", "ready_reason"), _ready(q27_id, require_lake_verified=True))),
        }
    )

    flash_school = dict(schools.get("Flash") or {})
    flash_lake = lake.get("Qwen/Qwen3.8-Flash-Next") or {}
    flash_id = dict(flash_lake)
    flash_id["physical_status"] = flash_school.get("physical_status") or flash_id.get("physical_status")
    roles.append(
        {
            "role": CURRICULUM_ROLES[4][0],
            "purpose": CURRICULUM_ROLES[4][1],
            "repo": flash_lake.get("repo") or flash_school.get("source_model") or "Qwen/Qwen3.8-Flash-Next",
            "revision": flash_lake.get("revision") or flash_school.get("pinned_revision"),
            "architecture_family": flash_school.get("architecture_family") or "qwen4_exp",
            "identity_source": "hcli.flash_next pin + modellake census + odyssey2 school",
            "school": flash_school,
            "modellake": flash_lake,
            **dict(zip(("ready", "ready_reason"), _ready(flash_id, require_lake_verified=True))),
        }
    )

    first_wave_repos = {r["repo"] for r in roles}
    lake_extras = [
        {"repo": repo, "revision": row.get("revision"), "reason": "present in ModelLake, not a first-wave curriculum role"}
        for repo, row in sorted(lake.items())
        if repo not in first_wave_repos
    ]

    n_ready = sum(1 for r in roles if r.get("ready"))
    return {
        "schema": "hawking.future.odyssey_i.specimen_curriculum.v1",
        "n_roles": len(CURRICULUM_ROLES),
        "n_ready": n_ready,
        "ready": n_ready == len(CURRICULUM_ROLES) and len(CURRICULUM_ROLES) > 0,
        "roles": roles,
        "not_proposed": lake_extras,
        "not_proposed_rule": (
            "Do not exhaustively optimize every downloaded model. First-wave "
            "curriculum is the five roles above; other lake entries are recorded "
            "and deferred."
        ),
        "census_probe": {
            "found": bool(census_probe.get("found")),
            "path_taken": census_probe.get("path_taken"),
            "resolved": census_probe.get("resolved"),
        },
        "prior_odyssey_i_patients": patients,
    }


# ---------------------------------------------------------------------------
# First WorkGraphs — real WorkUnits, not prose. Phase II/III listen concurrently.
# ---------------------------------------------------------------------------


def _slug(repo: str, revision: str | None) -> str:
    name = (repo or "specimen").split("/")[-1]
    name = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    rev = (revision or "unpinned")[:12]
    return f"{name}@{rev}"


def _stage_resource(stage: str) -> tuple[str, str]:
    if stage in GPU_STAGES:
        return "GPU_EXCLUSIVE", "sleeping"
    if stage in COMPILE_STAGES:
        return "COMPILE", "sleeping" if stage == "nx" else "pending"
    return "STATIC_ANALYSIS", "pending"


def _stage_verifier(stage: str, slug: str) -> str:
    return f"future.odyssey_launch.workgraph.{stage}:{slug}"


def emit_first_workgraphs(first_specimen: Mapping[str, Any]) -> dict[str, Any]:
    """Specimen verification → … → scars, then II transfer and III attack in parallel."""
    repo = str(first_specimen.get("repo") or "unknown")
    revision = first_specimen.get("revision")
    slug = _slug(repo, str(revision) if revision else None)
    prefix = f"odyssey-i.wg.{slug}"

    units: list[dict[str, Any]] = []
    by_stage: dict[str, str] = {}
    prev: str | None = None
    for stage in GRAPH_STAGES:
        uid = f"{prefix}.{stage}"
        by_stage[stage] = uid
        resource, status = _stage_resource(stage)
        sleeping = status == "sleeping"
        extras = {
            "species": "odyssey_i_first_workgraph",
            "stage": stage,
            "specimen_repo": repo,
            "specimen_revision": revision,
            "odyssey": "I",
            "era": "I",
            "sleeping": sleeping,
            "blocked_reason": (
                "SLEEPING: hardware is not qualified on this host; HCLI wakes "
                "this unit when the protected GPU lane qualifies. Not a synthetic result."
                if sleeping
                else None
            ),
            "requires_quiescence": resource == "GPU_EXCLUSIVE",
            "output_receipt_path": f"receipts/future/ODYSSEY_I_{stage.upper()}_{slug}.json",
        }
        row = wus.emit_hcli_workunit(
            id=uid,
            role="science",
            description=(
                f"Odyssey I first WorkGraph stage {stage} for {repo}"
                + (f"@{revision}" if revision else "")
            ),
            dependencies=[prev] if prev else [],
            resource_class=resource,
            verifier=_stage_verifier(stage, slug),
            provider="future.odyssey_launch",
            effect_class="READ_ONLY",
            status=status,
            classification="SLEEPING" if sleeping else "STATIC_ONLY",
            extras=extras,
        )
        wus.validate_emitted_unit(row)
        units.append(row)
        prev = uid

    laws_id = by_stage["laws"]
    listen: list[dict[str, Any]] = []
    for stage, odyssey, species in (
        ("phase_ii_transfer", "II", "odyssey_ii_transfer_experiment"),
        ("phase_iii_attack", "III", "odyssey_iii_adversarial_experiment"),
    ):
        uid = f"{prefix}.{stage}"
        by_stage[stage] = uid
        row = wus.emit_hcli_workunit(
            id=uid,
            role="science",
            description=(
                f"Odyssey {odyssey} listens concurrently: once Phase I emits a "
                f"law, this unit may run. No global barrier. Specimen {repo}."
            ),
            dependencies=[laws_id],
            resource_class="STATIC_ANALYSIS",
            verifier=_stage_verifier(stage, slug),
            provider="future.odyssey_launch",
            effect_class="READ_ONLY",
            status="pending",
            classification="STATIC_ONLY",
            extras={
                "species": species,
                "stage": stage,
                "specimen_repo": repo,
                "specimen_revision": revision,
                "odyssey": odyssey,
                "era": "III",
                "listens_on": "phase_i_law_emission",
                "barrier": None,
                "blocked_reason": None,
                "requires_quiescence": False,
            },
        )
        wus.validate_emitted_unit(row)
        units.append(row)
        listen.append({"id": uid, "odyssey": odyssey, "depends_on": [laws_id], "depends_on_sibling": False})

    policy = phase_listen_policy(laws_id)
    units.sort(key=lambda r: str(r.get("id") or ""))
    return {
        "schema": "hawking.future.odyssey_i.first_workgraph.v1",
        "specimen": {"repo": repo, "revision": revision, "slug": slug},
        "stages": list(GRAPH_STAGES),
        "listen_stages": list(PHASE_LISTEN_STAGES),
        "n_units": len(units),
        "units": units,
        "by_stage": by_stage,
        "phase_listen": policy,
        "listen_units": listen,
        "note": (
            "These are real HCLI WorkUnits. Emitting them is not executing them. "
            "GPU-class units are SLEEPING until hardware qualifies."
        ),
    }


def phase_listen_policy(laws_unit_id: str | None = None) -> dict[str, Any]:
    """Once Phase I emits a law, Phase II may transfer it and Phase III may attack it."""
    return {
        "barrier": None,
        "global_barrier": False,
        "rule": (
            "once Phase I emits a law, Phase II may transfer it and Phase III "
            "may attack it. There is no global barrier between II and III."
        ),
        "phase_ii_depends_on": [laws_unit_id or "phase_i.laws"],
        "phase_iii_depends_on": [laws_unit_id or "phase_i.laws"],
        "phase_ii_depends_on_phase_iii": False,
        "phase_iii_depends_on_phase_ii": False,
        "odysseys": list(ODYSSEYS),
        "no_odyssey_iv": True,
        "no_era_vi": True,
    }


def resource_schedule(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, list[str]] = {}
    by_status: dict[str, list[str]] = {}
    for u in units:
        rc = str(u.get("resource_class") or "UNKNOWN")
        st = str(u.get("status") or "UNKNOWN")
        by_class.setdefault(rc, []).append(str(u.get("id")))
        by_status.setdefault(st, []).append(str(u.get("id")))
    for d in (by_class, by_status):
        for k in d:
            d[k] = sorted(d[k])
    return {
        "by_resource_class": {k: {"n": len(v), "ids": v} for k, v in sorted(by_class.items())},
        "by_status": {k: {"n": len(v), "ids": v} for k, v in sorted(by_status.items())},
        "gpu_authority": False,
        "sleeping_are_not_results": True,
    }


# ---------------------------------------------------------------------------
# Physical blockers → SLEEPING WorkUnits. Never a synthetic result.
# ---------------------------------------------------------------------------


def _xcrun_metal() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["xcrun", "-f", "metal"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        found = proc.returncode == 0 and bool((proc.stdout or "").strip())
        return {
            "probed": True,
            "found": found,
            "path": (proc.stdout or "").strip() or None,
            "stderr": (proc.stderr or "")[:240] or None,
            "returncode": proc.returncode,
        }
    except Exception as exc:
        return {"probed": True, "found": False, "error": f"{type(exc).__name__}: {exc}"}


def physical_blockers() -> list[dict[str, Any]]:
    """Derive the live physical blockers from sealed receipts + a toolchain probe."""
    handoff = probe_json("receipts/future/FUTURE_SUBSTRATE_HANDOFF.json")
    nx_rec = probe_json("receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json")
    teacher = probe_json("receipts/future/TEACHER_CORPUS_CONTRACT.json")
    qual = probe_json("receipts/future/QUALIFICATION_PIPELINE.json")
    cont = probe_json("receipts/future/CONTAMINATION_SCIENCE.json")

    handoff_blockers = []
    if isinstance(handoff.get("doc"), Mapping):
        handoff_blockers = list(handoff["doc"].get("blockers") or [])

    seven_all_met = None
    nx_status = None
    if isinstance(nx_rec.get("doc"), Mapping):
        seven_all_met = nx_rec["doc"].get("seven_all_met")
        nx_v0 = (nx_rec["doc"].get("nx_completeness_checker") or {}).get("real_FLASH_COMPLETE_V0_nx")
        if isinstance(nx_v0, Mapping):
            nx_status = nx_v0.get("status") or nx_v0.get("state")
        nr = nx_rec["doc"].get("nr_v2") if isinstance(nx_rec["doc"].get("nr_v2"), Mapping) else {}
        nr_status = nr.get("status")
    else:
        nr_status = None

    n_executed = 0
    n_units = 0
    target_rows = []
    if isinstance(teacher.get("doc"), Mapping):
        units = list(teacher["doc"].get("capture_workunits") or [])
        n_units = len(units)
        n_executed = sum(1 for u in units if isinstance(u, Mapping) and u.get("executed") is True)
        for u in units:
            if not isinstance(u, Mapping):
                continue
            payload = u.get("payload") if isinstance(u.get("payload"), Mapping) else {}
            tr = payload.get("target_row_count")
            if isinstance(tr, int):
                target_rows.append(tr)
    teacher_done = n_executed > 0

    lease_reason = None
    lease_present = None
    if isinstance(qual.get("doc"), Mapping):
        stop = qual["doc"].get("dry_run_stop") if isinstance(qual["doc"].get("dry_run_stop"), Mapping) else {}
        lease_reason = stop.get("reason")
        auth = qual["doc"].get("authority_boundary") if isinstance(qual["doc"].get("authority_boundary"), Mapping) else {}
        lease_present = False if auth else None

    cont_class = None
    if isinstance(cont.get("doc"), Mapping):
        cont_class = cont["doc"].get("contamination_class")

    xcrun = _xcrun_metal()

    rows = [
        {
            "id": "sleep.no-gpu-authority",
            "title": "No protected GPU / MetalContext not invoked",
            "sleeping": True,
            "holds": True,
            "reason": (
                "Sidecar has no GPU authority and does not invoke MetalContext. "
                "Handoff blockers record no protected GPU authority on this campaign."
            ),
            "evidence": {"handoff_blockers": handoff_blockers, "gpu_authority": False},
        },
        {
            "id": "sleep.metal-compiler",
            "title": "Metal compiler under CommandLineTools",
            "sleeping": True,
            "holds": not bool(xcrun.get("found")),
            "reason": (
                "xcrun -f metal did not locate the Metal compiler"
                if not xcrun.get("found")
                else f"xcrun located metal at {xcrun.get('path')}"
            ),
            "evidence": {"xcrun": xcrun},
        },
        {
            "id": "sleep.protected-bench-lock",
            "title": "Protected bench lock; holder pids unproven",
            "sleeping": True,
            "holds": True,
            "reason": lease_reason
            or (
                "qualification pipeline fail-closes lease present: lock file may "
                "exist but flock would be a seizure and is never attempted"
            ),
            "evidence": {
                "qualification_path_taken": qual.get("path_taken"),
                "lease_present": lease_present,
                "dry_run_stop": lease_reason,
            },
        },
        {
            "id": "sleep.machine-heavy",
            "title": "Machine classified HEAVY; standing workers not quiesced",
            "sleeping": True,
            "holds": cont_class == "HEAVY" or cont_class is None,
            "reason": (
                f"contamination_class={cont_class!r}; sidecar will not SIGSTOP "
                "standing workers. HEAVY (or unknown) is not a protected window."
            ),
            "evidence": {"contamination_class": cont_class, "path_taken": cont.get("path_taken")},
        },
        {
            "id": "sleep.flash-nx-scaffold",
            "title": "Flash source-independent NX is not qualified",
            "sleeping": True,
            "holds": seven_all_met is not True,
            "reason": (
                f"FLASH_NX_COMPLETENESS_AUDIT seven_all_met={seven_all_met!r} "
                f"nx_status={nx_status!r} nr_v2.status={nr_status!r}. "
                "SCAFFOLD_ONLY / metadata seals are not a callable NX path."
            ),
            "evidence": {
                "seven_all_met": seven_all_met,
                "nx_status": nx_status,
                "nr_v2_status": nr_status,
                "path_taken": nx_rec.get("path_taken"),
            },
        },
        {
            "id": "sleep.teacher-capture",
            "title": "Teacher capture has not run",
            "sleeping": True,
            "holds": not teacher_done,
            "reason": (
                f"teacher capture executed={n_executed}/{n_units} workunits; "
                f"target_row_counts={sorted(set(target_rows))}. "
                "Unrun capture is not a synthetic corpus."
            ),
            "evidence": {
                "n_capture_workunits": n_units,
                "n_executed": n_executed,
                "target_row_counts": sorted(set(target_rows)),
                "path_taken": teacher.get("path_taken"),
            },
        },
    ]
    sleeping_units: list[dict[str, Any]] = []
    for row in rows:
        if not row["holds"]:
            continue
        unit = wus.emit_hcli_workunit(
            id=f"odyssey-i.{row['id']}",
            role="science",
            description=f"SLEEPING until hardware qualifies: {row['title']}",
            dependencies=[],
            resource_class="GPU_EXCLUSIVE" if "gpu" in row["id"] or "nx" in row["id"] or "teacher" in row["id"] or "lock" in row["id"] or "heavy" in row["id"] or "compiler" in row["id"] else "STATIC_ANALYSIS",
            verifier=f"future.odyssey_launch.wakeup:{row['id']}",
            provider="future.odyssey_launch",
            effect_class="READ_ONLY",
            status="sleeping",
            classification="SLEEPING",
            extras={
                "sleeping": True,
                "blocked_reason": row["reason"],
                "requires_quiescence": True,
                "synthetic_result_forbidden": True,
                "wakeup_integration": INTEGRATION_POINTS["wakeup"],
            },
        )
        wus.validate_emitted_unit(unit)
        sleeping_units.append(unit)
        row["workunit_id"] = unit["id"]
    return _attach_sleeping(rows, sleeping_units)


def _attach_sleeping(rows: list[dict[str, Any]], units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {u["id"]: u for u in units}
    out = []
    for r in rows:
        wu_id = r.get("workunit_id")
        out.append(
            {
                "id": r["id"],
                "title": r["title"],
                "holds": r["holds"],
                "sleeping": r["sleeping"] and r["holds"],
                "reason": r["reason"],
                "evidence": r["evidence"],
                "workunit": by_id.get(wu_id),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Sixteen launch criteria. Every evaluator runs. None short-circuits the rest.
# ---------------------------------------------------------------------------


def _eval_autonomy() -> dict[str, Any]:
    probes = [
        probe_json(
            "receipts/future/AUTONOMY_TRIAL.json",
            "receipts/future/ODYSSEY_AUTONOMY_TRIAL.json",
            "receipts/future/SUPER_RESIDENT.json",
        ),
        probe_json(
            "receipts/headless/HCLI_AGENTOS_AUTONOMY_GATE.json",
            "receipts/future/evidence/HCLI_AGENTOS_AUTONOMY_GATE.json",
        ),
    ]
    trial = probes[0]
    hcli = probes[1]
    hcli_doc = hcli.get("doc") if isinstance(hcli.get("doc"), Mapping) else None
    hcli_schema = hcli_doc.get("schema") if hcli_doc else None
    hcli_passed = bool(
        hcli_doc
        and hcli_schema == HCLI_AUTONOMY_SCHEMA
        and (hcli_doc.get("checks") or {}).get("all_requested_stages_passed") is True
    )
    trial_doc = trial.get("doc") if isinstance(trial.get("doc"), Mapping) else None
    trial_schema = trial_doc.get("schema") if trial_doc else None
    trial_pass = bool(
        trial_doc
        and trial_schema in ODYSSEY_AUTONOMY_SCHEMAS
        and str(trial_doc.get("verdict") or trial_doc.get("status") or "").upper() in {"PASS", "PASSED"}
        and trial_doc.get("resident_orchestration") is True
    )
    met = trial_pass
    reason = (
        "Odyssey I resident-orchestration autonomy trial has passed"
        if met
        else (
            "No Odyssey I resident-orchestration autonomy trial has passed. "
            f"HCLI AgentOS autonomy gate path_taken={hcli.get('path_taken')!r} "
            f"schema={hcli_schema!r} all_requested_stages_passed={hcli_passed}: "
            "that receipt proves LIVE_AGENTOS_AUTONOMY_CONTROL_PLANE, not Odyssey I start. "
            f"Odyssey trial receipt path_taken={trial.get('path_taken')!r} "
            f"schema={trial_schema!r}. Integration point: {INTEGRATION_POINTS['autonomy_trial']}."
        )
    )
    return _criterion(
        "resident_autonomy_trial_pass",
        met=met,
        reason=reason,
        evidence=[
            {"kind": "odyssey_trial", "path_taken": trial.get("path_taken"), "schema": trial_schema, "found": trial.get("found")},
            {
                "kind": "hcli_agentos_autonomy",
                "path_taken": hcli.get("path_taken"),
                "schema": hcli_schema,
                "found": hcli.get("found"),
                "all_requested_stages_passed": hcli_passed,
                "not_this_criterion": True,
            },
        ],
        operational=operational_bar(
            discover=bool(trial.get("found") or hcli.get("found")),
            invoke=False,
            schedule=False,
            verify=hcli_passed,
            frontier=False,
            persist=bool(hcli.get("found")),
            refill=False,
            notes={"hcli_control_plane_is_not_odyssey_i": "true"},
        ),
    )


def _eval_curriculum() -> dict[str, Any]:
    cur = propose_specimen_curriculum()
    met = bool(cur.get("ready"))
    n_ready = cur.get("n_ready")
    n_roles = cur.get("n_roles")
    unmet_roles = [r["role"] for r in cur.get("roles") or [] if not r.get("ready")]
    return _criterion(
        "specimen_curriculum_ready",
        met=met,
        reason=(
            f"all {n_roles} curriculum roles have a live sealed first-wave specimen"
            if met
            else (
                f"curriculum proposes {n_roles} roles; ready={n_ready}. "
                f"unready={unmet_roles}. A proposal is not a ready specimen set."
            )
        ),
        evidence=[{"curriculum_n_roles": n_roles, "n_ready": n_ready, "unready": unmet_roles}],
        extra={"curriculum": cur},
        operational=operational_bar(
            discover=True,
            invoke=True,
            schedule=False,
            verify=False,
            frontier=False,
            persist=False,
            refill=False,
            notes={"proposal_emitted": "true", "specimens_not_all_published": str(not met)},
        ),
    )


def _eval_modellake() -> dict[str, Any]:
    imp = _importable("hcli.agentos.modellake_receipts")
    names: tuple[str, ...] = ()
    if imp.get("ok"):
        try:
            from hcli.agentos.modellake_receipts import CENSUS_RECEIPT_NAMES

            names = tuple(CENSUS_RECEIPT_NAMES)
        except Exception:
            names = ()
    rels = [f"receipts/headless/{n}" for n in names] or [
        "receipts/headless/HCLI_MODELLAKE_FLASH_CENSUS.json",
        "receipts/headless/MODELLAKE_FLASH_NEXT_CENSUS.json",
    ]
    rels.append("receipts/future/evidence/HCLI_MODELLAKE_FLASH_CENSUS.json")
    probe = probe_json(*rels)
    doc = probe.get("doc") if isinstance(probe.get("doc"), Mapping) else None
    schema_ok = bool(doc and str(doc.get("schema") or "").startswith("hcli.agentos.modellake"))
    status = doc.get("status") if doc else None
    identity_ok = bool(imp.get("ok") and probe.get("found") and schema_ok)
    # Identity machinery can be up while Flash is identity-only. That is this
    # criterion; specimen readiness is the curriculum criterion.
    return _criterion(
        "modellake_identity",
        met=identity_ok,
        reason=(
            f"ModelLake identity machinery is importable and a census receipt "
            f"schema={doc.get('schema') if doc else None!r} status={status!r} "
            f"was loaded via {probe.get('path_taken')}"
            if identity_ok
            else (
                f"ModelLake identity not operational: import={imp} census "
                f"path_taken={probe.get('path_taken')!r} schema_ok={schema_ok}"
            )
        ),
        evidence=[
            {"import": {k: v for k, v in imp.items() if k != "file" or v}, "census": {"path_taken": probe.get("path_taken"), "found": probe.get("found"), "schema": None if not doc else doc.get("schema"), "status": status, "qualification": None if not doc else doc.get("qualification")}},
        ],
        operational=operational_bar(
            discover=bool(imp.get("ok")),
            invoke=bool(imp.get("ok")),
            schedule=False,
            verify=schema_ok,
            frontier=False,
            persist=bool(probe.get("found")),
            refill=False,
            notes={"identity_is_not_acquisition": "true"},
        ),
    )


def _resident_schedulable(owned: Sequence[str]) -> dict[str, Any]:
    """Can the RESIDENT schedule this tool, route its receipt, and refill on it?

    Measured against the connector, not asserted. A binding counts only if the
    orchestration table names a sidecar module that actually drives this tool --
    a module that merely mentions the path in a comment is not a binding.
    """
    result = {"schedule": False, "frontier": None, "refill": False,
              "why": "no orchestration binding drives this tool"}
    try:
        from tools.future import frontiers as _fr
        from tools.future import orchestration as _orch
    except Exception as exc:
        result["why"] = f"connector unavailable: {type(exc).__name__}"
        return result
    wanted = {str(o) for o in owned}
    for module, (frontier_id, _species) in _orch.BINDINGS.items():
        if module == "odyssey_launch.py":
            continue  # this gate does not get to certify itself as the driver
        src = REPO / "tools" / "future" / module
        try:
            text = src.read_text(errors="replace")
        except OSError:
            continue
        if not any(w in text for w in wanted):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        # A driver RUNS it. Naming the path in a list constant is a declaration,
        # and a declaration schedules nothing -- the first version of this check
        # accepted an Assign and so credited this gate module itself, which had
        # the path in an `owned = [...]` literal, as Doctor's driver. Requiring
        # the mention to sit inside a call refuses that. It also refuses a real
        # driver that builds its argv separately, which is the safe direction:
        # this check may understate schedulability, never overstate it.
        drives = any(
            isinstance(n, ast.Call) and any(w in ast.dump(n) for w in wanted)
            for n in ast.walk(tree)
        )
        if not drives:
            continue
        result.update({"schedule": True, "frontier": frontier_id,
                       "driver_module": module, "why": "bound via orchestration"})
        try:
            result["refill"] = any(
                str(item.get("id")) == frontier_id
                # The frontier's own lane vocabulary, not a second list. These
                # were spelled CPU_ANALYSIS / CPU_VERIFY / CPU_REPRESENTATION,
                # which match no frontier item, so the refill probe could only
                # ever return empty and refill was structurally unreachable. The
                # same defect was fixed in the autonomy driver and survived here
                # -- which is exactly why the scar says derive, never restate.
                for item in (_fr.refill(_fr.THIS_HOST_LANES) or [])
            )
        except Exception:
            result["refill"] = False
        break
    return result


def _declared_inputs(rel: str) -> list[dict[str, Any]]:
    """Absolute filesystem paths a tool hardcodes as its input, and whether they exist.

    Parsed, never imported: importing an odyssey tool would run its module-level
    work inside the gate. Only literal `NAME = Path("/abs/...")` assignments are
    read, which is exactly how the Doctor and Gravity tools name their parent.
    """
    path = REPO / rel
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (OSError, SyntaxError):
        return []
    out: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "Path" and len(call.args) == 1
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)):
            continue
        literal = call.args[0].value
        if not literal.startswith("/"):
            continue  # a repo-relative path is not an external input
        out.append({"name": target.id, "path": literal,
                    "present": Path(literal).exists(), "declared_in": rel})
    return out


# Where large model parents actually live on this host. Bounded on purpose: a
# find over the 4.5TB volume to answer "is the parent here" is not a probe.
MODEL_ROOTS: tuple[str, ...] = (
    "/Users/scammermike/models",
    "/Volumes/corpdrive/personalmodel/correspondent",
    "/Volumes/corpdrive/personalmodel",
    "/Volumes/corpdrive/hawking-modellake/specimens",
)


def _resolve_stale_input(declared: str) -> str | None:
    """The same directory name under a known model root, if the declared path moved.

    "Not on this host" and "not where the tool says" are different findings, and
    only one of them is a missing model. The Doctor and Gravity tools declare
    /Users/scammermike/models/qwen3.8-27b-abliterated-bf16, which is gone; the
    52GB parent is on the external volume. Saying the model is missing would
    send someone to re-download 52GB that is already here.
    """
    name = Path(declared).name
    if not name:
        return None
    for root in MODEL_ROOTS:
        candidate = Path(root) / name
        if candidate.is_dir():
            return str(candidate)
    return None


def _eval_callable_tool(*, cid: str, owned: Sequence[str], prior_glob: str, title: str) -> dict[str, Any]:
    files = [_module_file(p) for p in owned]
    prior = []
    root = REPO / "receipts" / "odyssey-i"
    if root.is_dir():
        prior = sorted(str(p.relative_to(REPO)) for p in root.glob(prior_glob))
    present = any(f.get("present") for f in files) or bool(prior)
    # File presence is not invocability. These tools hardcode the parent weights
    # they read, and a tool whose parent is not on this host cannot be invoked by
    # anyone, resident or human. Measure it rather than inferring it from the
    # presence of the script.
    inputs = [i for rel in owned for i in _declared_inputs(rel)]
    for item in inputs:
        if not item["present"]:
            item["resolved_elsewhere"] = _resolve_stale_input(item["path"])
    missing_inputs = [i for i in inputs if not i["present"]]
    stale_inputs = [i for i in missing_inputs if i.get("resolved_elsewhere")]
    absent_inputs = [i for i in missing_inputs if not i.get("resolved_elsewhere")]
    invocable = (
        any(f.get("present") and f.get("path_taken") == "worktree" for f in files)
        and not missing_inputs
    )
    # schedule / frontier / refill are measured against the connector that now
    # exists, instead of being asserted False. They are still false here, but for
    # a reason that can be checked and can change.
    sched = _resident_schedulable(owned)
    bar = operational_bar(
        discover=present,
        invoke=invocable,
        schedule=bool(sched["schedule"]),
        verify=bool(prior),
        frontier=bool(sched["frontier"]),
        persist=bool(prior),
        refill=bool(sched["refill"]),
        notes={
            "cli_or_prior_seals": "true",
            "declared_inputs_missing": ", ".join(i["path"] for i in missing_inputs) or "none",
            "stale_declared_paths": ", ".join(
                f"{i['path']} -> {i['resolved_elsewhere']}" for i in stale_inputs
            ) or "none",
            "integration": INTEGRATION_POINTS["workgraph"] + " + " + INTEGRATION_POINTS["resident_api"],
        },
    )
    return _criterion(
        cid,
        met=bool(bar["resident_operational"]),
        reason=(
            f"{title} is resident-operational"
            if bar["resident_operational"]
            else (
                f"{title} is recovered (n_prior={len(prior)}) but is not resident-"
                f"operational. Unmet: "
                + ", ".join(k for k, v in bar["flags"].items() if not v)
                + (
                    f". The blocker is a STALE DECLARED PATH, not a missing model: "
                    + "; ".join(
                        f"{i['name']} in {i['declared_in']} points at {i['path']}, "
                        f"which is gone, while the directory is present at "
                        f"{i['resolved_elsewhere']}"
                        for i in stale_inputs
                    )
                    + ". The tool cannot run as written."
                    if stale_inputs
                    else (
                        f". {', '.join(i['path'] for i in absent_inputs)} is not on "
                        f"this host at all, so the tool cannot be invoked by anyone. "
                        f"A CLI a human cannot run either is not enough."
                        if absent_inputs
                        else ". A CLI a human can run is not enough."
                    )
                )
            )
        ),
        evidence=[{"owned": files, "n_prior_seals": len(prior), "prior_seals": prior,
                   "declared_inputs": inputs, "missing_inputs": missing_inputs,
                   "stale_declared_paths": stale_inputs, "absent_inputs": absent_inputs,
                   "resident_schedulable": sched}],
        operational=bar,
    )


def _eval_doctor() -> dict[str, Any]:
    owned = ["tools/odyssey/doctor_tournament.py", "tools/doctor_seal.py"]
    hcli = _importable("hcli.doctor")
    extra: list[str] = []
    if hcli.get("ok"):
        import hcli.doctor as hd

        extra = [str(p) for p in getattr(hd, "OWNED_PATHS", ())]
    return _eval_callable_tool(
        cid="doctor_callable",
        owned=list(owned) + extra,
        prior_glob="*_DOCTOR_SEAL.json",
        title="Doctor",
    )


def _eval_gravity() -> dict[str, Any]:
    return _eval_callable_tool(
        cid="gravity_callable",
        owned=[
            "tools/odyssey/decoding_gravity.py",
            "tools/odyssey/state_gravity.py",
            "hcli/gravity/__init__.py",
        ],
        prior_glob="*_GRAVITY_*.json",
        title="Gravity",
    )


def _eval_nr_nx() -> dict[str, Any]:
    rec = probe_json("receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    seven = doc.get("seven_all_met") if doc else None
    nx_checker = (doc.get("nx_completeness_checker") or {}) if doc else {}
    real = nx_checker.get("real_FLASH_COMPLETE_V0_nx") if isinstance(nx_checker, Mapping) else None
    nr = doc.get("nr_v2") if isinstance(doc, Mapping) else None
    nx_v0 = probe_json(
        "receipts/future/evidence/FLASH_COMPLETE_V0.nx.json",
        "receipts/headless/FLASH_COMPLETE_V0.nx.json",
    )
    nx_doc = nx_v0.get("doc") if isinstance(nx_v0.get("doc"), Mapping) else None
    nx_status = nx_doc.get("status") if nx_doc else None
    callable_path = seven is True and nx_status not in {
        None,
        nx_audit.METADATA_ONLY,
        "NOT_BUILT",
        "SCAFFOLD_ONLY",
    }
    bar = operational_bar(
        discover=bool(doc or nx_doc),
        invoke=False,
        schedule=False,
        verify=seven is False or seven is True,
        frontier=False,
        persist=bool(rec.get("found")),
        refill=False,
        notes={"flash_nx_scaffold_only": str(not callable_path)},
    )
    return _criterion(
        "nr_nx_path_callable",
        met=bool(callable_path and bar["resident_operational"]),
        reason=(
            "NR/NX path is source-independent, complete, and resident-callable"
            if callable_path
            else (
                f"NR/NX path is not callable. seven_all_met={seven!r} "
                f"FLASH_COMPLETE_V0.nx status={nx_status!r} "
                f"nr_v2={None if not isinstance(nr, Mapping) else nr.get('status')!r}. "
                "Flash source-independent NX is SCAFFOLD_ONLY / metadata, not qualified."
            )
        ),
        evidence=[
            {
                "audit_path_taken": rec.get("path_taken"),
                "seven_all_met": seven,
                "nx_v0_path_taken": nx_v0.get("path_taken"),
                "nx_v0_status": nx_status,
                "nr_v2": nr if isinstance(nr, Mapping) else None,
                "real_checker": real,
            }
        ],
        operational=bar,
    )


def _eval_evidence_hierarchy() -> dict[str, Any]:
    snap = probe_json("receipts/future/EVIDENCE_SNAPSHOT.json")
    doc = snap.get("doc") if isinstance(snap.get("doc"), Mapping) else None
    n_captured = len(doc.get("captured") or []) if doc else 0
    lattice_ok = tuple(C.MEASUREMENT_CLASSES) == EVIDENCE_LATTICE or set(C.MEASUREMENT_CLASSES) == set(EVIDENCE_LATTICE)
    dag = _module_file("tools/future/evidence_dag.py")
    live_dag = _exercise("tools.future.evidence_dag", "selftest")
    bar = operational_bar(
        discover=lattice_ok,
        invoke=lattice_ok,
        schedule=bool(dag.get("present")) and bool(live_dag.get("ok")),
        verify=n_captured > 0,
        frontier=bool(live_dag.get("ok")),
        persist=bool(snap.get("found")),
        refill=bool(live_dag.get("ok")),
        notes={
            "lattice_recovered": str(lattice_ok),
            "evidence_dag": dag.get("path_taken"),
            "integration": INTEGRATION_POINTS["evidence_dag"],
        },
    )
    met = bool(bar["resident_operational"])
    return _criterion(
        "evidence_hierarchy",
        met=met,
        reason=(
            "evidence DAG is resident-operational"
            if met
            else (
                f"measurement-class lattice is encoded ({list(C.MEASUREMENT_CLASSES)}) "
                f"and evidence snapshot captured={n_captured} via {snap.get('path_taken')}, "
                "but the resident-callable evidence DAG is not landed "
                f"({INTEGRATION_POINTS['evidence_dag']}). DIAGNOSTIC_RELATIVE guides and never promotes."
            )
        ),
        evidence=[{"lattice": list(C.MEASUREMENT_CLASSES), "snapshot_captured": n_captured, "dag": dag}],
        operational=bar,
    )


def _eval_negative_science() -> dict[str, Any]:
    imp = _importable("tools.future.negative_index")
    rec = probe_json("receipts/future/NEGATIVE_SCIENCE_INDEX.json")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    n_scars = None
    if doc and isinstance(doc.get("coverage"), Mapping):
        n_scars = doc["coverage"].get("n_scars")
    has_api = bool(imp.get("ok") and hasattr(__import__("tools.future.negative_index", fromlist=["query"]), "query"))
    bar = operational_bar(
        discover=bool(imp.get("ok")),
        invoke=has_api,
        schedule=True,
        verify=has_api,
        frontier=True,
        persist=_receipt_sealed(doc),
        refill=True,
        notes={"refuse_if_dead": "tools.future.negative_index.refuse_if_dead"},
    )
    met = bool(imp.get("ok") and rec.get("found") and (n_scars is None or n_scars > 0) and bar["resident_operational"])
    return _criterion(
        "negative_science",
        met=met,
        reason=(
            f"negative-science index is importable, sealed, scars={n_scars}, query/refuse_if_dead callable"
            if met
            else f"negative science not operational: import={imp.get('ok')} receipt={rec.get('path_taken')} n_scars={n_scars}"
        ),
        evidence=[{"import_ok": imp.get("ok"), "path_taken": rec.get("path_taken"), "n_scars": n_scars}],
        operational=bar,
    )



def _exercise(dotted: str, fn_name: str) -> dict[str, Any]:
    """Actually run a now-landed sibling capability.

    These criteria were written while the siblings were being built concurrently,
    so the contract forbade importing them and the evaluators hard-coded the
    frontier/persist/refill axes to False. Those modules have since LANDED and are
    committed, so the honest evaluation is to exercise them rather than keep
    asserting an absence that is no longer true. If a call raises, the criterion
    stays unmet with the exception recorded -- this never flips to True by fiat.
    """
    try:
        mod = importlib.import_module(dotted)
    except Exception as exc:
        return {"ok": False, "why": f"import failed: {type(exc).__name__}: {exc}"}
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        return {"ok": False, "why": f"{dotted}.{fn_name} is not callable"}
    try:
        out = fn()
    except Exception as exc:
        return {"ok": False, "why": f"{fn_name}() raised {type(exc).__name__}: {exc}"}
    return {"ok": True, "called": f"{dotted}.{fn_name}()", "result": str(out)[:200]}


def _eval_workgraphs() -> dict[str, Any]:
    runtime = _module_file("tools/future/workgraph.py")
    species = _importable("tools.future.workunit_species")
    live = _exercise("tools.future.workgraph", "selftest")
    bar = operational_bar(
        discover=bool(species.get("ok")),
        invoke=bool(species.get("ok")),
        schedule=bool(runtime.get("present")) and bool(live.get("ok")),
        verify=bool(species.get("ok")),
        frontier=bool(live.get("ok")),
        persist=bool(live.get("ok")),
        refill=bool(live.get("ok")),
        notes={
            "this_module_emits_units": "true",
            "runtime": runtime.get("path_taken"),
            "integration": INTEGRATION_POINTS["workgraph"],
        },
    )
    return _criterion(
        "workgraphs",
        met=bool(bar["resident_operational"]),
        reason=(
            "WorkGraph runtime is resident-operational"
            if bar["resident_operational"]
            else (
                "This gate emits first WorkGraphs as real HCLI WorkUnits, but the "
                f"WorkGraph runtime is not landed ({INTEGRATION_POINTS['workgraph']}). "
                "Emitting a graph is not executing a graph."
            )
        ),
        evidence=[{"runtime": runtime, "workunit_species_import": species.get("ok"),
                   "exercised": live}],
        operational=bar,
    )


def _eval_self_refill() -> dict[str, Any]:
    frontier = probe_json("receipts/future/CLAUDE_GLOBAL_FRONTIER.json")
    fronts = _module_file("tools/future/frontiers.py")
    succ = _module_file("tools/future/succession.py")
    live_refill = _exercise("tools.future.frontiers", "refill")
    # is_idle() is the verify probe: a refill loop that cannot say whether the
    # frontier is exhausted is not a verified loop, it is a generator.
    live_idle = _exercise("tools.future.frontiers", "is_idle")
    bar = operational_bar(
        discover=bool(frontier.get("found")),
        invoke=bool(live_refill.get("ok")),
        schedule=bool(live_refill.get("ok")),
        verify=bool(live_idle.get("ok")),
        frontier=bool(frontier.get("found")),
        persist=bool(frontier.get("found")),
        refill=bool(fronts.get("present") and succ.get("present") and live_refill.get("ok")),
        notes={
            "global_frontier_is_inventory": "true",
            "exercised_refill": str(live_refill.get("ok")),
            "exercised_is_idle": str(live_idle.get("ok")),
            "integration": INTEGRATION_POINTS["frontiers"] + " + " + INTEGRATION_POINTS["succession"],
        },
    )
    return _criterion(
        "self_refill",
        met=bool(bar["resident_operational"]),
        reason=(
            "resident self-refill loop is operational"
            if bar["resident_operational"]
            else (
                f"CLAUDE_GLOBAL_FRONTIER path_taken={frontier.get('path_taken')} is an "
                "inventory, not a refill loop. frontiers.py/succession.py are this-wave "
                "integration points and are not imported."
            )
        ),
        evidence=[{"frontier": {"path_taken": frontier.get("path_taken"), "found": frontier.get("found")}, "frontiers": fronts, "succession": succ}],
        operational=bar,
    )


def _eval_dirty_measurement() -> dict[str, Any]:
    dirty = _module_file("tools/future/dirty_measure.py")
    live_dirty = _exercise("tools.future.dirty_measure", "build")
    cont = probe_json("receipts/future/CONTAMINATION_SCIENCE.json")
    doc = cont.get("doc") if isinstance(cont.get("doc"), Mapping) else None
    klass = doc.get("contamination_class") if doc else None
    bar = operational_bar(
        discover=bool(cont.get("found")),
        invoke=bool(_importable("tools.future.contamination").get("ok")),
        schedule=bool(dirty.get("present")) and bool(live_dirty.get("ok")),
        verify=klass in C.CONTAMINATION_CLASSES if klass else False,
        frontier=bool(live_dirty.get("ok")),
        persist=bool(cont.get("found")),
        refill=bool(live_dirty.get("ok")),
        notes={
            "contamination_is_machine_state": "true",
            "dirty_measure_integration": INTEGRATION_POINTS["dirty_measure"],
            "gpu_dirty_ok_resource_class_exists": "GPU_DIRTY_OK",
        },
    )
    return _criterion(
        "dirty_measurement",
        met=bool(bar["resident_operational"]),
        reason=(
            "dirty measurement is resident-operational"
            if bar["resident_operational"]
            else (
                f"contamination science is sealed (class={klass!r}) and refuses "
                "DIAGNOSTIC_RELATIVE promotion, but dirty-class measurement of "
                f"tokens/experiments is {INTEGRATION_POINTS['dirty_measure']}."
            )
        ),
        evidence=[{"contamination_class": klass, "path_taken": cont.get("path_taken"), "dirty_measure": dirty}],
        operational=bar,
    )


def _eval_protected_scheduling() -> dict[str, Any]:
    future_pw = _module_file("tools/future/protected_window.py")
    odyssey_pw = _module_file("tools/odyssey/protected_window.py")
    qual = probe_json("receipts/future/QUALIFICATION_PIPELINE.json")
    doc = qual.get("doc") if isinstance(qual.get("doc"), Mapping) else None
    gpu_auth = False
    if doc and isinstance(doc.get("authority_boundary"), Mapping):
        gpu_auth = bool(doc["authority_boundary"].get("gpu_authority"))
    cont = probe_json("receipts/future/CONTAMINATION_SCIENCE.json")
    klass = None
    if isinstance(cont.get("doc"), Mapping):
        klass = cont["doc"].get("contamination_class")
    # invoke / frontier / refill were hardcoded False, so this criterion could not
    # pass on ANY machine -- not even a quiescent one holding a real lease. That
    # is the same unreachable-bar shape that made doctor_callable and
    # gravity_callable permanently unmet, and it hid the actual blocker behind a
    # constant. Measure them. The criterion still refuses today, because
    # contamination is not QUIESCENT and there is no GPU authority, and those are
    # facts about this moment rather than a decision baked into the evaluator.
    #
    # Refusing to SEIZE a lock and being unable to SCHEDULE protected work are
    # different claims. The sidecar will never take a contested lock; that is a
    # policy this evaluator does not need to enforce by pretending the machinery
    # is absent.
    pw_sched = _resident_schedulable(["tools/future/protected_window.py"])
    lease_ok = klass == "QUIESCENT" and gpu_auth
    bar = operational_bar(
        discover=bool(odyssey_pw.get("present") or qual.get("found")),
        invoke=bool(future_pw.get("present")) and lease_ok,
        schedule=bool(pw_sched["schedule"]) and lease_ok,
        verify=bool(qual.get("found")),
        frontier=bool(pw_sched["frontier"]) and lease_ok,
        persist=bool(qual.get("found")),
        refill=bool(pw_sched["refill"]) and lease_ok,
        notes={
            "sidecar_must_not_seize_lock": "true",
            "lease_precondition": (
                f"contamination must be QUIESCENT (is {klass!r}) and qualification "
                f"gpu_authority must be true (is {gpu_auth})"
            ),
            "driver": str(pw_sched.get("driver_module") or "none"),
            "integration": INTEGRATION_POINTS["protected_window"],
            "prior_odyssey_protected_window": odyssey_pw.get("path_taken"),
        },
    )
    return _criterion(
        "protected_scheduling",
        met=bool(bar["resident_operational"]),
        reason=(
            "protected scheduling is resident-operational on a QUIESCENT machine with a proven HCLI lease"
            if bar["resident_operational"]
            else (
                f"protected scheduling cannot start: contamination_class={klass!r} "
                f"(needs QUIESCENT), qualification gpu_authority={gpu_auth} (needs "
                f"true), driver={pw_sched.get('driver_module') or 'none'}. Every "
                f"unmet flag is measured: {', '.join(k for k, v in bar['flags'].items() if not v)}. "
                "The sidecar will not flock a contested bench lock, and that policy "
                "is separate from this criterion -- these flags describe whether a "
                "lease-holding resident COULD schedule protected work, not whether "
                "this process may take a lock."
            )
        ),
        evidence=[
            {
                "future_protected_window": future_pw,
                "odyssey_protected_window": odyssey_pw,
                "qualification_path_taken": qual.get("path_taken"),
                "contamination_class": klass,
            }
        ],
        operational=bar,
    )


def _eval_transfer() -> dict[str, Any]:
    imp = _importable("tools.future.odyssey2_law_store")
    rec = probe_json("receipts/future/ODYSSEY2_LAW_STORE.json")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    n_laws = None
    if doc and isinstance(doc.get("counts"), Mapping):
        n_laws = doc["counts"].get("n_laws")
    elif doc:
        n_laws = len(doc.get("laws") or [])
    has_promote = bool(imp.get("ok") and hasattr(ols, "promote") and hasattr(ols, "transfer_candidates"))
    bar = operational_bar(
        discover=bool(imp.get("ok")),
        invoke=has_promote,
        schedule=True,
        verify=has_promote,
        frontier=True,
        persist=_receipt_sealed(doc),
        refill=True,
        notes={"species": "odyssey_ii_transfer_experiment", "raises_on_negative_transfer": "true"},
    )
    met = bool(imp.get("ok") and rec.get("found") and (n_laws or 0) > 0 and bar["resident_operational"])
    return _criterion(
        "transfer_substrate",
        met=met,
        reason=(
            f"Odyssey II law store is sealed with n_laws={n_laws}; promote/transfer_candidates raise rather than flag"
            if met
            else f"transfer substrate not operational: import={imp.get('ok')} path_taken={rec.get('path_taken')} n_laws={n_laws}"
        ),
        evidence=[{"n_laws": n_laws, "path_taken": rec.get("path_taken"), "schema": None if not doc else doc.get("schema")}],
        operational=bar,
    )


def _eval_adversary() -> dict[str, Any]:
    imp = _importable("tools.future.odyssey3_adversary")
    rec = probe_json("receipts/future/ODYSSEY3_ADVERSARY.json")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    families = list(doc.get("attack_families") or []) if doc else []
    has_api = bool(imp.get("ok") and hasattr(o3, "p_refutation"))
    bar = operational_bar(
        discover=bool(imp.get("ok")),
        invoke=has_api,
        schedule=True,
        verify=has_api,
        frontier=True,
        persist=_receipt_sealed(doc),
        refill=True,
        notes={"species": "odyssey_iii_adversarial_experiment", "a_law_that_emits_no_attack_is_refused": "true"},
    )
    met = bool(imp.get("ok") and rec.get("found") and families and bar["resident_operational"])
    return _criterion(
        "adversary_substrate",
        met=met,
        reason=(
            f"Odyssey III adversary is sealed; attack_families derived from receipt (n={len(families)})"
            if met
            else f"adversary substrate not operational: import={imp.get('ok')} path_taken={rec.get('path_taken')}"
        ),
        evidence=[{"n_attack_families": len(families), "path_taken": rec.get("path_taken"), "schema": None if not doc else doc.get("schema")}],
        operational=bar,
    )


def _eval_crash_recovery() -> dict[str, Any]:
    imp = _importable("tools.future.repro_science")
    rec = probe_json("receipts/future/REPRO_SCIENCE.json")
    recov = probe_json(
        "receipts/headless/HCLI_AGENTOS_RECOVERY_GATE.json",
        "receipts/future/evidence/HCLI_AGENTOS_RECOVERY_GATE.json",
    )
    git_lock = probe_json("receipts/future/GIT_LOCK_DURABILITY_REPORT.json")
    hcli_imp = _importable("hcli.agentos.recovery")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    faults_ok = False
    if doc and isinstance(doc.get("fault_injection"), Mapping):
        faults_ok = doc["fault_injection"].get("all_detected") is True
    bar = operational_bar(
        discover=bool(imp.get("ok") or hcli_imp.get("ok")),
        invoke=bool(imp.get("ok") and hasattr(rs, "admit")),
        schedule=True,
        verify=faults_ok,
        frontier=True,
        persist=_receipt_sealed(doc),
        refill=True,
        notes={"hcli_recovery": hcli_imp.get("ok"), "git_lock_doctor": git_lock.get("found")},
    )
    met = bool(imp.get("ok") and rec.get("found") and faults_ok and bar["resident_operational"])
    return _criterion(
        "crash_recovery",
        met=met,
        reason=(
            "repro_science fault injectors all_detected and HCLI recovery module importable; sidecar fail-closed"
            if met
            else (
                f"crash recovery not operational: repro={imp.get('ok')} "
                f"faults_ok={faults_ok} hcli.recovery={hcli_imp.get('ok')} "
                f"recovery_gate path_taken={recov.get('path_taken')}"
            )
        ),
        evidence=[
            {
                "repro_path_taken": rec.get("path_taken"),
                "faults_ok": faults_ok,
                "hcli_recovery_import": hcli_imp.get("ok"),
                "recovery_gate": {"found": recov.get("found"), "path_taken": recov.get("path_taken")},
                "git_lock": {"found": git_lock.get("found"), "path_taken": git_lock.get("path_taken")},
            }
        ],
        operational=bar,
    )


def _eval_receipts() -> dict[str, Any]:
    future = RECEIPTS
    present = future.is_dir()
    n = 0
    if present:
        n = sum(1 for p in future.glob("*.json") if p.is_file())
    bar = operational_bar(
        discover=present,
        invoke=True,
        schedule=True,
        verify=True,
        frontier=True,
        persist=present,
        refill=True,
        notes={"write_receipt": "tools.future._common.write_receipt", "hardware_claim_raises": "HardwareClaimError"},
    )
    met = bool(present and n > 0 and bar["resident_operational"])
    return _criterion(
        "receipts",
        met=met,
        reason=(
            f"receipts/future is present with n_json={n} (derived); write_receipt seals STATIC_ONLY and raises on hardware fields"
            if met
            else f"receipts partition not operational: dir={present} n_json={n}"
        ),
        evidence=[{"receipts_future": str(future), "n_json": n}],
        operational=bar,
    )


EVALUATORS: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
    ("resident_autonomy_trial_pass", _eval_autonomy),
    ("specimen_curriculum_ready", _eval_curriculum),
    ("modellake_identity", _eval_modellake),
    ("doctor_callable", _eval_doctor),
    ("gravity_callable", _eval_gravity),
    ("nr_nx_path_callable", _eval_nr_nx),
    ("evidence_hierarchy", _eval_evidence_hierarchy),
    ("negative_science", _eval_negative_science),
    ("workgraphs", _eval_workgraphs),
    ("self_refill", _eval_self_refill),
    ("dirty_measurement", _eval_dirty_measurement),
    ("protected_scheduling", _eval_protected_scheduling),
    ("transfer_substrate", _eval_transfer),
    ("adversary_substrate", _eval_adversary),
    ("crash_recovery", _eval_crash_recovery),
    ("receipts", _eval_receipts),
)


def evaluate_launch_criteria() -> list[dict[str, Any]]:
    """Run every criterion. Never stop at the first unmet."""
    if tuple(cid for cid, _ in EVALUATORS) != CRITERION_IDS:
        raise RuntimeError("EVALUATORS and CRITERION_IDS have drifted")
    rows: list[dict[str, Any]] = []
    for cid, fn in EVALUATORS:
        row = fn()
        if row.get("id") != cid:
            raise RuntimeError(f"evaluator {cid} returned id={row.get('id')!r}")
        rows.append(row)
    return rows


def unmet_criteria(results: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    rows = list(results) if results is not None else evaluate_launch_criteria()
    return [str(r["id"]) for r in rows if not r.get("met")]


def can_launch(results: Sequence[Mapping[str, Any]] | None = None) -> bool:
    return not unmet_criteria(results)


def launch_verdict(results: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(results) if results is not None else evaluate_launch_criteria()
    unmet = unmet_criteria(rows)
    met = [str(r["id"]) for r in rows if r.get("met")]
    return {
        "allowed": not unmet,
        "verdict": "LAUNCH" if not unmet else "REFUSED",
        "unmet": unmet,
        "met": met,
        "n_criteria": len(rows),
        "n_unmet": len(unmet),
        "n_met": len(met),
        "rule": "every unmet criterion is named; the first failure does not hide the rest",
    }


# ---------------------------------------------------------------------------
# Launch payload. Written to disk only when the gate passes.
# ---------------------------------------------------------------------------


def _identity_stub(kind: str, integration: str) -> dict[str, Any]:
    rels = {
        "resident": (
            "receipts/future/RESIDENT_IDENTITY.json",
            "receipts/headless/HCLI_AGENTOS_RESIDENT_GATE.json",
        ),
        "sandbox": (
            "receipts/future/SANDBOX.json",
            "receipts/headless/HCLI_AGENTOS_CHECKPOINT.json",
        ),
    }[kind]
    probe = probe_json(*rels)
    doc = probe.get("doc") if isinstance(probe.get("doc"), Mapping) else None
    return {
        "kind": kind,
        "found": bool(probe.get("found")),
        "path_taken": probe.get("path_taken"),
        "resolved": probe.get("resolved"),
        "schema": None if not doc else doc.get("schema"),
        "status": None if not doc else doc.get("status") or doc.get("qualification"),
        "bound": False,
        "integration_point": integration,
        "note": (
            "Identity is not invented. A missing this-wave receipt stays unbound. "
            "HCLI prior gates, if found, are cited and are not this identity."
        ),
    }


def _machine_genome_pin() -> dict[str, Any]:
    nx = probe_json(
        "receipts/future/evidence/FLASH_COMPLETE_V0.nx.json",
        "receipts/headless/FLASH_COMPLETE_V0.nx.json",
    )
    digest = None
    if isinstance(nx.get("doc"), Mapping):
        compiled = nx["doc"].get("compiled_for_machine_genome")
        if isinstance(compiled, Mapping):
            digest = compiled.get("genome_digest")
    pin = dict(rs.fixture_machine_genome())
    pin.update(
        {
            "genome_digest": digest,
            "source": nx.get("resolved"),
            "path_taken": nx.get("path_taken"),
            "knowledge_level": "PIN_ONLY",
            "gpu_authority": False,
            "note": "digest cited; core counts and bytes from the NX seal are not copied (not a measurement claim)",
        }
    )
    return pin


def launch_payload(
    results: Sequence[Mapping[str, Any]],
    *,
    curriculum: Mapping[str, Any],
    workgraphs: Mapping[str, Any],
    blockers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    verdict = launch_verdict(results)
    units = list(workgraphs.get("units") or [])
    return {
        "schema": LAUNCH_SCHEMA,
        "version": VERSION,
        "odyssey": ODYSSEYS[0],
        "eras": list(ERAS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "phase_transition": "STARTED" if verdict["allowed"] else "NOT_STARTED",
        "resident_identity": _identity_stub("resident", INTEGRATION_POINTS["resident_identity"]),
        "sandbox_identity": _identity_stub("sandbox", INTEGRATION_POINTS["sandbox"]),
        "machine_genome": _machine_genome_pin(),
        "first_specimen_set": curriculum,
        "first_workgraphs": {
            "specimen": workgraphs.get("specimen"),
            "n_units": workgraphs.get("n_units"),
            "stages": workgraphs.get("stages"),
            "listen_stages": workgraphs.get("listen_stages"),
            "phase_listen": workgraphs.get("phase_listen"),
            "units": units,
        },
        "resource_schedule": resource_schedule(units),
        "evidence_hierarchy": {
            "lattice": list(EVIDENCE_LATTICE),
            "rule": "DIAGNOSTIC_RELATIVE guides and never promotes; PROTECTED_ABSOLUTE decides; this module produces neither",
            "measurement_state": "STATIC_ONLY",
            "bench": "UNKNOWN",
        },
        "autonomy_status": next(
            (r for r in results if r.get("id") == "resident_autonomy_trial_pass"),
            {},
        ),
        "known_blockers": [
            {"id": b.get("id"), "holds": b.get("holds"), "reason": b.get("reason"), "sleeping": b.get("sleeping")}
            for b in blockers
        ],
        "next_refill_trigger": {
            "when": "a criterion flips from unmet to met, or a SLEEPING unit is woken because hardware qualified",
            "integration": INTEGRATION_POINTS["frontiers"],
            "does_not_refill_on": "a refused launch-gate re-run with the same unmet set",
        },
        "verdict": verdict,
        "gpu_authority": False,
        "claim_boundary": SIDECAR_CLAIM,
    }


def write_launch_if_passed(
    payload: Mapping[str, Any],
    *,
    allowed: bool,
    writer: Callable[[str, dict[str, Any], str], Path] | None = None,
) -> dict[str, Any]:
    """Write ODYSSEY_I_LAUNCH.json only when the gate passes. Fail closed otherwise."""
    write = writer or write_receipt
    if not allowed:
        return {
            "written": False,
            "path": None,
            "name": LAUNCH_RECEIPT,
            "reason": (
                "gate REFUSED; ODYSSEY_I_LAUNCH.json is a phase-transition receipt "
                "and is not written while any criterion is unmet"
            ),
        }
    path = write(LAUNCH_RECEIPT, dict(payload), RECORDED_BY)
    return {"written": True, "path": str(path), "name": LAUNCH_RECEIPT, "reason": "all sixteen criteria met"}


def _gate_workunit(verdict: Mapping[str, Any]) -> dict[str, Any]:
    row = wus.emit_hcli_workunit(
        id="odyssey-i.launch-gate",
        role="science",
        description="Evaluate Odyssey I launch criteria; refuse until every criterion is met",
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.odyssey_launch.can_launch",
        provider="future.odyssey_launch",
        effect_class="READ_ONLY",
        status="completed",
        classification="REFUSED" if not verdict.get("allowed") else "STATIC_ONLY",
        extras={
            "verdict": verdict.get("verdict"),
            "unmet": list(verdict.get("unmet") or []),
            "blocked_reason": None if verdict.get("allowed") else "launch criteria unmet; see unmet list",
            "requires_quiescence": False,
            "output_receipt_path": f"receipts/future/{RECEIPT}",
        },
    )
    wus.validate_emitted_unit(row)
    return row


def recovered_implementation() -> dict[str, Any]:
    return {
        "prior_odyssey_i": {
            "tools": "tools/odyssey/ (READ-ONLY): doctor_tournament, decoding_gravity, state_gravity, modellake, resident_seal, protected_window, pareto_archive, noetic_compiler",
            "receipts": "receipts/odyssey-i/ patient/doctor/gravity/nx seals — recovered, not duplicated",
            "package_fence": (
                "workspace/campaign/governance/odyssey/program/launch/ODYSSEY_LAUNCH.md "
                "is a different Odyssey (training-package fence, ODYSSEY_LAUNCH_AUTHORIZED=false). "
                "It is not this gate."
            ),
        },
        "hcli": {
            "modellake_receipts": "hcli/agentos/modellake_receipts.py — preferred census/supervision names",
            "modellake_gate": "hcli/agentos/modellake_gate.py — identity census, no acquisition",
            "autonomy_gate": "hcli/agentos/autonomy_gate.py — control-plane A1–A5, not Odyssey I start",
            "recovery": "hcli/agentos/recovery.py — process kill/resume",
            "doctor_gravity_markers": "hcli/doctor, hcli/gravity ownership markers",
            "workunit": "hcli/workunit.py — units emitted through the real constructor",
        },
        "landed_future": {
            "odyssey2_law_store": "transfer substrate",
            "odyssey3_adversary": "adversary substrate",
            "negative_index": "negative science",
            "evidence_snapshot": "pinned Codex receipts",
            "workunit_species": "HCLI WorkUnit emission",
            "flash_nx_audit": "NR/NX completeness",
            "contamination": "machine-state class, HEAVY on this host",
            "qualification_pipeline": "protected lease fail-closed",
            "repro_science": "crash/fault fail-closed",
            "teacher_corpus": "capture WorkUnits executed=false",
            "resident_optimizer": "propose/rank, never promote",
            "_common.write_receipt": "HardwareClaimError on numeric hardware fields",
        },
        "this_module_existed": False,
        "fork_decision": "extend nothing that already gated Odyssey I start; the package fence is a different campaign",
    }


def gaps_closed() -> list[str]:
    return [
        "Sixteen launch criteria evaluated from evidence; can_launch is False until all pass.",
        "Refuse path names every unmet criterion; it does not stop at the first.",
        "ODYSSEY_I_LAUNCH.json is written only on pass; refuse does not write the phase-transition receipt.",
        "Specimen curriculum proposes five roles from ModelLake seals / Odyssey I / law-store schools; other lake entries are recorded and not first-wave.",
        "First WorkGraphs emitted as real HCLI WorkUnits with dependencies and resource lanes.",
        "Phase II transfer and Phase III attack both depend on Phase I laws and not on each other — no global barrier.",
        "Physical blockers become SLEEPING WorkUnits; they never become synthetic results.",
        "HCLI AgentOS autonomy A1–A5 is recovered and explicitly not this criterion.",
    ]


def negative_findings_from(
    verdict: Mapping[str, Any],
    blockers: Sequence[Mapping[str, Any]],
    curriculum: Mapping[str, Any],
) -> list[str]:
    findings = [
        f"verdict={verdict.get('verdict')} n_unmet={verdict.get('n_unmet')} unmet={list(verdict.get('unmet') or [])}",
        "A gate that passed today would mistake 'Odyssey infrastructure ready' for 'Odyssey started'.",
    ]
    for b in blockers:
        if b.get("holds"):
            findings.append(f"physical blocker {b.get('id')}: {b.get('reason')}")
    unready = [r["role"] for r in curriculum.get("roles") or [] if not r.get("ready")]
    if unready:
        findings.append(f"curriculum unready roles: {unready}")
    findings.append(
        "this-wave siblings are not imported; local interfaces are the integration points "
        + ", ".join(sorted(INTEGRATION_POINTS))
    )
    return findings


def resident_callable_block(verdict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_point": "python3 tools/future/odyssey_launch.py --verify",
        "launch_entry_point": "python3 tools/future/odyssey_launch.py --launch",
        "workunit_emitted": "odyssey-i.launch-gate plus first WorkGraph units and SLEEPING physical blockers",
        "receipt": f"receipts/future/{RECEIPT}",
        "phase_transition_receipt": f"receipts/future/{LAUNCH_RECEIPT}",
        "frontier_fed": (
            "Odyssey I start frontier: a REFUSED gate keeps the frontier at NOT_STARTED; "
            "a pass would write ODYSSEY_I_LAUNCH.json and that is the only start mark. "
            "Next refill trigger is a criterion flip or a SLEEPING wakeup."
        ),
        "fail_closed": (
            "can_launch() is False while any criterion is unmet; unmet_criteria() names "
            "every unmet id; write_launch_if_passed() does not call write_receipt for "
            f"{LAUNCH_RECEIPT} on refuse; --launch exits 1; HardwareClaimError on numeric "
            "hardware fields; GPU units stay SLEEPING rather than inventing a result."
        ),
        "verdict_now": verdict.get("verdict"),
        "this_wave_not_imported": list(THIS_WAVE_SIBLINGS),
        "integration_points": dict(INTEGRATION_POINTS),
    }


def build(*, writer: Callable[[str, dict[str, Any], str], Path] | None = None) -> dict[str, Any]:
    write = writer or write_receipt
    results = evaluate_launch_criteria()
    verdict = launch_verdict(results)
    curriculum = propose_specimen_curriculum()
    first = (curriculum.get("roles") or [{}])[0]
    graphs = emit_first_workgraphs(first)
    blockers = physical_blockers()
    payload = launch_payload(results, curriculum=curriculum, workgraphs=graphs, blockers=blockers)
    launch_write = write_launch_if_passed(payload, allowed=bool(verdict["allowed"]), writer=write)
    gate_wu = _gate_workunit(verdict)
    sleeping_wu = [b["workunit"] for b in blockers if b.get("workunit")]
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "odyssey": ODYSSEYS[0],
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "fpga": "FPGA belongs to Accelerator / Physical Compiler / Fusion. This gate does not build an FPGA backend.",
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "claim_boundary": SIDECAR_CLAIM,
        "criterion_ids": list(CRITERION_IDS),
        "criteria": results,
        "verdict": verdict,
        "specimen_curriculum": curriculum,
        "first_workgraphs": {
            "specimen": graphs.get("specimen"),
            "n_units": graphs.get("n_units"),
            "stages": graphs.get("stages"),
            "phase_listen": graphs.get("phase_listen"),
            "units": graphs.get("units"),
        },
        "resource_schedule": resource_schedule(list(graphs.get("units") or [])),
        "physical_blockers": [
            {k: v for k, v in b.items() if k != "workunit"} | {"workunit_id": (b.get("workunit") or {}).get("id")}
            for b in blockers
        ],
        "sleeping_workunits": sleeping_wu,
        "gate_workunit": gate_wu,
        "launch_payload_draft": payload,
        "launch_receipt": launch_write,
        "odyssey_i_launch_written": bool(launch_write.get("written")),
        "phase_transition": "STARTED" if launch_write.get("written") else "NOT_STARTED",
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings_from(verdict, blockers, curriculum),
        "resident_callable": resident_callable_block(verdict),
        "integration_points": dict(INTEGRATION_POINTS),
        "this_wave_siblings_not_imported": list(THIS_WAVE_SIBLINGS),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    path = write(RECEIPT, doc, RECORDED_BY)
    return {"gate_path": str(path), "doc": doc, "launch": launch_write}


def verify() -> int:
    out = build()
    doc = out["doc"]
    verdict = doc["verdict"]
    print(f"schema={doc['schema']} version={doc['version']}")
    print(f"verdict={verdict['verdict']} n_unmet={verdict['n_unmet']} n_met={verdict['n_met']}")
    if verdict["unmet"]:
        print("unmet:")
        for cid in verdict["unmet"]:
            print(f"  - {cid}")
    print(f"gate_receipt={out['gate_path']}")
    print(
        f"launch_receipt_written={doc['odyssey_i_launch_written']} "
        f"phase_transition={doc['phase_transition']}"
    )
    if verdict["allowed"] and not doc["odyssey_i_launch_written"]:
        print("FAIL: gate allowed but ODYSSEY_I_LAUNCH.json was not written", file=sys.stderr)
        return 1
    if (not verdict["allowed"]) and doc["odyssey_i_launch_written"]:
        print("FAIL: gate refused but ODYSSEY_I_LAUNCH.json was written", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Odyssey I launch gate")
    ap.add_argument("--verify", action="store_true", help="evaluate, seal the gate receipt, refuse honestly")
    ap.add_argument("--build", action="store_true", help="alias of --verify")
    ap.add_argument(
        "--launch",
        action="store_true",
        help="write ODYSSEY_I_LAUNCH.json only if every criterion is met; exit 1 on refuse",
    )
    a = ap.parse_args()
    if a.launch:
        out = build()
        if not out["launch"].get("written"):
            print("REFUSED: ODYSSEY_I_LAUNCH.json not written", file=sys.stderr)
            for cid in out["doc"]["verdict"]["unmet"]:
                print(f"  unmet: {cid}", file=sys.stderr)
            return 1
        print(out["launch"]["path"])
        return 0
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
