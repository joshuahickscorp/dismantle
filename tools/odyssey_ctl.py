#!/usr/bin/env python3
"""Odyssey-I patient controller — queue, harvest, packets, frontier.

Queue skeleton follows tools/ascent_controller.py (load/save/value/loop shape)
but uses a new state file and a patient/work schema. Does not touch Genesis
ascent state or tools/odyssey/ (training-data Odyssey).

    python3 tools/odyssey_ctl.py --self-check
    python3 tools/odyssey_ctl.py status
    python3 tools/odyssey_ctl.py queue
    python3 tools/odyssey_ctl.py value
    python3 tools/odyssey_ctl.py harvest
    python3 tools/odyssey_ctl.py packet O005
    python3 tools/odyssey_ctl.py admit <slug> <est_gib>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import doctor_seal  # noqa: E402
import worker_gate  # noqa: E402

ODYSSEY = REPO / "workspace" / "campaign" / "odyssey"
STATE = ODYSSEY / "ODYSSEY_STATE.json"
LEDGER = ODYSSEY / "ODYSSEY.md"
SCHEMA_PATH = ODYSSEY / "patient_packet_schema.json"
PATIENTS_DIR = ODYSSEY / "patients"
RECEIPT_DIR = REPO / "receipts" / "odyssey-i"
ESCALATIONS = ODYSSEY / "OPUS_ESCALATIONS.jsonl"
RULEBASE = ODYSSEY / "GRAVITY_RULEBASE.json"
TRANSFER = ODYSSEY / "TRANSFER_MATRIX.json"
NEGATIVE = ODYSSEY / "NEGATIVE_SCIENCE.json"
A3B_RECON = REPO / "receipts" / "ascent-2026-08-18" / "A3B_RECON.json"
GROK_TASKS = Path.home() / ".claude-grok" / "tasks"
RECLAIM = TOOLS / "reclaim_safe.sh"

DISK_FLOOR_GIB = 15.0
DISK_WARN_GIB = 40.0
SCHEMA = "hawking.odyssey.controller.v1"

STATES = (
    "READY", "RUNNING", "BLOCKED", "LANDED",
    "VERIFYING", "VERIFIED", "REFUTED", "ARCHIVED",
)
EVIDENCE = (
    "VERIFIED", "MEASURED", "DERIVED", "INFERRED",
    "HYPOTHESIS", "SPECULATIVE", "REFUTED", "STALE", "UNKNOWN",
)
PACKET_SECTIONS = (
    "identity", "architecture", "doctor", "tabula", "routing",
    "representation", "execution", "gravity", "nx", "transfer", "next",
)
MOE_PATIENTS = {
    "O003", "O005", "O006", "O007", "O008",
    "O010", "O011", "O012", "O013",
}

# §22 high-value kinds get higher `info`. Costs are coarse units, not money.
SEED_WORK = [
    {"id": "A1", "oxx": "O005", "title": "route/state map (instrument router)",
     "status": "READY", "info": 10, "wall_cost": 2, "gpu_cost": 1, "opus_cost": 0,
     "kind": "router-sensitivity"},
    {"id": "A2", "oxx": "O005", "title": "baseline TPS via external runtime",
     "status": "READY", "info": 4, "wall_cost": 1, "gpu_cost": 1, "opus_cost": 0,
     "kind": "architecture-first"},
    {"id": "A3", "oxx": "O005", "title": "per-organ/per-expert sensitivity map",
     "status": "READY", "info": 10, "wall_cost": 2, "gpu_cost": 1, "opus_cost": 0,
     "kind": "representation-discriminator"},
    {"id": "A5", "oxx": "O001", "title": "SSM organ bucket + state-vs-KV",
     "status": "READY", "info": 8, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0,
     "kind": "architecture-first"},
    {"id": "A6", "oxx": "O000", "title": "HF-token unblock O000/O002/O004",
     "status": "BLOCKED", "info": 5, "wall_cost": 0, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O003", "oxx": "O003", "title": "acquire Kimi-VL-A3B",
     "status": "READY", "info": 3, "wall_cost": 3, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O006", "oxx": "O006", "title": "acquire Qwen3-VL sibling (transfer ctrl)",
     "status": "READY", "info": 4, "wall_cost": 3, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O007", "oxx": "O007", "title": "acquire Kimi-Linear-48B-A3B",
     "status": "READY", "info": 3, "wall_cost": 4, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O008", "oxx": "O008", "title": "acquire Jamba-Mini-1.5",
     "status": "READY", "info": 3, "wall_cost": 3, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O009", "oxx": "O009", "title": "acquire Qwen2.5-72B",
     "status": "READY", "info": 2, "wall_cost": 5, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O010", "oxx": "O010", "title": "acquire GLM-4.5-Air",
     "status": "READY", "info": 4, "wall_cost": 5, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O011", "oxx": "O011", "title": "DSV4F legacy replay from receipts",
     "status": "READY", "info": 6, "wall_cost": 2, "gpu_cost": 1, "opus_cost": 0,
     "kind": "false-win-discovery"},
    {"id": "ACQ-O012", "oxx": "O012", "title": "acquire GLM-4.5 full (partial-residency)",
     "status": "READY", "info": 2, "wall_cost": 6, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
    {"id": "ACQ-O013", "oxx": "O013", "title": "acquire Kimi-K3 streamed capstone",
     "status": "READY", "info": 2, "wall_cost": 8, "gpu_cost": 0, "opus_cost": 0,
     "kind": "acquisition"},
]

ROW_RE = re.compile(
    r"^\|\s*(O\d{3})\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|"
)
OXX_RE = re.compile(r"\bO(\d{3})\b", re.I)
SLUG_OXX_RE = re.compile(r"odyssey-o(\d{3})", re.I)
JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
COMPLETION_RE = re.compile(
    r"(?:\*\*Completion report\*\*|#{1,3}\s+Completion report)\s*(.*)",
    re.S | re.I,
)
RESULT_RE = re.compile(r"^RESULT\s*:", re.M)
FALLBACK_PATIENTS = [
    ("O000", "gemma-3-1b-it", "tiny dense (compiler lab)",
     "google/gemma-3-1b-it", "HF-gated", "BLOCKED-auth", "—"),
    ("O001", "Falcon-H1-7B-Instruct", "small hybrid attn+Mamba",
     "tiiuae/Falcon-H1-7B-Instruct", "open", "on-disk", "CENSUS✓"),
    ("O002", "gemma-3-4b-it", "small dense multimodal",
     "google/gemma-3-4b-it", "HF-gated", "BLOCKED-auth", "—"),
    ("O003", "Kimi-VL-A3B-Instruct", "small multimodal MoE",
     "moonshotai/Kimi-VL-A3B-Instruct", "open", "queued", "—"),
    ("O004", "Mistral-Small-3.1-24B", "medium dense multimodal",
     "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "HF-gated", "BLOCKED-auth", "—"),
    ("O005", "Qwen3-30B-A3B", "small-active MoE",
     "Qwen/Qwen3-30B-A3B", "open", "on-disk", "CENSUS✓"),
    ("O006", "Qwen3-VL-30B-A3B", "multimodal MoE sibling",
     "Qwen/Qwen3-VL-30B-A3B-Instruct", "open", "queued (transfer ctrl)", "—"),
    ("O007", "Kimi-Linear-48B-A3B", "linear-attn MoE (KDA+MLA)",
     "moonshotai/Kimi-Linear-48B-A3B-Instruct", "open", "queued", "—"),
    ("O008", "Jamba-Mini-1.5", "Mamba+attn MoE",
     "ai21labs/AI21-Jamba-Mini-1.5", "license?", "queued", "—"),
    ("O009", "Qwen2.5-72B-Instruct", "large dense",
     "Qwen/Qwen2.5-72B-Instruct", "open", "queued (large)", "—"),
    ("O010", "GLM-4.5-Air", "106B/12B MoE",
     "zai-org/GLM-4.5-Air", "open", "queued (~220GB)", "—"),
    ("O011", "DSV4F", "mandatory legacy replay",
     "reconstruct from receipts", "local", "queued (control)", "—"),
    ("O012", "GLM-4.5 full", "355B/32B very-large MoE",
     "zai-org/GLM-4.5", "open", "queued (partial-residency)", "—"),
    ("O013", "Kimi-K3", "frontier 2.8T/104B streamed",
     "moonshotai/Kimi-K3", "open", "queued (streamed capstone)", "—"),
]


def strip_md(s: str) -> str:
    return (s or "").replace("**", "").strip()


def evidence_class(label) -> str | None:
    if label is None:
        return None
    text = str(label)
    for ev in EVIDENCE:
        if ev in text:
            return ev
    return None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def machine_snapshot() -> dict:
    """Same resource-governor shape as ascent_controller.machine_snapshot."""
    try:
        from agentos.machine_state import clean_box_ok, snapshot

        snap = snapshot()
        ok, why = clean_box_ok(snap, min_free_gib=DISK_FLOOR_GIB)
        snap["clean_box_ok"], snap["clean_box_reason"] = ok, why
        return snap
    except Exception as exc:  # pragma: no cover - environment drift
        st = os.statvfs(REPO)
        gib = st.f_bavail * st.f_frsize / 1024**3
        return {
            "disk_free_gib": round(gib, 1),
            "clean_box_ok": gib >= DISK_FLOOR_GIB,
            "clean_box_reason": f"machine_state unavailable ({exc})",
        }


def reclaim_if_tight(snap: dict) -> None:
    if (snap.get("disk_free_gib") or 999) < DISK_WARN_GIB and RECLAIM.is_file():
        subprocess.run(["bash", str(RECLAIM)], cwd=REPO, check=False)


def value(work: dict) -> float:
    """§22 proxy: expected reusable-compiler-info / (wall+gpu+opus cost).

    Ordering only. No fake precision.
    """
    info = float(work.get("info") or 0.0)
    cost = (
        float(work.get("wall_cost") or 0.0)
        + float(work.get("gpu_cost") or 0.0)
        + float(work.get("opus_cost") or 0.0)
    )
    return info / max(cost, 0.1)


def map_state(ledger: str) -> str:
    raw = strip_md(ledger)
    low = raw.lower()
    if low.startswith("blocked"):
        return "BLOCKED"
    if raw.upper() in STATES:
        return raw.upper()
    # on-disk and queued both have READY work (measure / acquire)
    return "READY"


def norm_phase(phase: str) -> str:
    p = strip_md(phase).replace("✓", "").replace("✔", "").strip()
    return "" if p in {"", "—", "-"} else p


def norm_oxx(s: str) -> str:
    s = strip_md(s).upper()
    if s.startswith("O"):
        s = s[1:]
    return f"O{int(s):03d}"


def parse_ledger_rows() -> list[tuple]:
    if LEDGER.is_file():
        rows = []
        for line in LEDGER.read_text().splitlines():
            m = ROW_RE.match(line)
            if not m:
                continue
            oxx = m.group(1)
            if oxx.startswith("O") and oxx[1:].isdigit():
                rows.append(tuple(strip_md(g) for g in m.groups()))
        if len(rows) >= 14:
            return rows
    return list(FALLBACK_PATIENTS)


def patient_record(row: tuple) -> dict:
    oxx, model, klass, source, gate, ledger, phase = row
    state = map_state(ledger)
    on_disk = strip_md(ledger).lower().startswith("on-disk")
    blocked = state == "BLOCKED"
    return {
        "oxx": oxx,
        "model": model,
        "class": klass,
        "source": source,
        "gate": gate,
        "ledger": ledger,
        "state": state,
        "phase": norm_phase(phase),
        "on_disk": on_disk,
        "blocked_reason": "HF-gated / BLOCKED-auth" if blocked else None,
        "_evidence": "VERIFIED (ODYSSEY.md patient table)",
    }


def empty_state() -> dict:
    return {
        "schema": SCHEMA,
        "patients": [patient_record(r) for r in parse_ledger_rows()],
        "work": [dict(w) for w in SEED_WORK],
        "history": [],
        "harvested": [],
        "metrics": {
            "opus_calls": 1,
            "grok_lanes_bootstrap": 3,
            "gpu_seconds": 0,
            "gpu_owner": "none",
            "_evidence": "INFERRED (ODYSSEY.md §72 session bootstrap)",
        },
    }


def load_state() -> dict:
    if STATE.is_file():
        st = read_json(STATE)
        if not st.get("patients"):
            st["patients"] = empty_state()["patients"]
        if not st.get("work"):
            st["work"] = [dict(w) for w in SEED_WORK]
        return st
    st = empty_state()
    save_state(st)
    return st


def save_state(state: dict) -> None:
    write_json(STATE, state)


def ensure_state() -> dict:
    return load_state()


def packet_path(oxx: str) -> Path:
    return PATIENTS_DIR / oxx / f"ODYSSEY_PATIENT_{oxx}.json"


def census_path(oxx: str) -> Path:
    return PATIENTS_DIR / oxx / "census.json"


def load_packet(oxx: str) -> dict | None:
    p = packet_path(oxx)
    return read_json(p) if p.is_file() else None


def section_defaults(oxx: str, meta: dict) -> dict:
    kind = "moe" if oxx in MOE_PATIENTS else (
        "hybrid" if "hybrid" in (meta.get("class") or "").lower() else "dense"
    )
    return {
        "identity": {
            "source_repo": meta.get("source") or "",
            "revision": None,
            "content_hashes": None,
            "tokenizer": None,
            "source_precision": None,
            "source_type": None,
            "license": None,
            "model_family": None,
            "_evidence": "UNKNOWN",
        },
        "architecture": {
            "total_params": None,
            "active_params": None,
            "kind": kind,
            "layers": None,
            "attention": None,
            "state_ssm": None,
            "moe_topology": None,
            "experts": None,
            "experts_per_tok": None,
            "shared_experts": None,
            "modality": None,
            "context": None,
            "census_ref": None,
            "_evidence": "UNKNOWN",
        },
        "doctor": {
            "capability_vector": None,
            "controls": None,
            "long_context_result": None,
            "coding_reasoning": None,
            "behavioral_profile": None,
            "known_weaknesses": None,
            "blind_spots": None,
            "fast_doctor_seal_ref": None,
            "full_doctor_seal_ref": None,
            "_evidence": "UNKNOWN",
        },
        "tabula": {
            "status": "N/A",
            "behavioral_identity": None,
            "drift_monitor": None,
            "_evidence": "UNKNOWN",
        },
        "routing": {
            "entropy": None,
            "expert_frequency": None,
            "transitions": None,
            "co_occurrence": None,
            "hot_set": None,
            "cold_set": None,
            "route_predictability": None,
            "P(E_t|E_t-1)": None,
            "_evidence": "UNKNOWN",
        },
        "representation": {
            "source_bytes": None,
            "best_stored_bpw_eq": None,
            "active_bpw_eq": None,
            "metadata_bytes": None,
            "corrections": None,
            "reconstruction": None,
            "per_organ_sensitivity": None,
            "per_expert_sensitivity": None,
            "_evidence": "UNKNOWN",
        },
        "execution": {
            "token_ns": None,
            "tps": None,
            "ttft": None,
            "prefill": None,
            "long_context_slope": None,
            "active_learned_bytes_per_token": None,
            "dram_per_token": None,
            "cache": None,
            "state": None,
            "route_overhead": None,
            "baseline_runtime": None,
            "baseline_tps": None,
            "_evidence": "UNKNOWN",
        },
        "gravity": {
            "tried_mechanisms": [],
            "wins": [],
            "kills": [],
            "architecture_specific_findings": [],
            "_evidence": "UNKNOWN",
        },
        "nx": {
            "primitive_set": None,
            "machine_lowering": None,
            "kernel_bindings": None,
            "fallback_count": None,
            "best_preliminary_nx": None,
            "nx_hash": None,
            "_evidence": "UNKNOWN",
        },
        "transfer": {
            "inherited_rules": [],
            "unchanged": [],
            "retuned": [],
            "failed": [],
            "harmful": [],
            "_evidence": "UNKNOWN",
        },
        "next": [],
    }


def patient_meta(oxx: str, state: dict | None = None) -> dict:
    st = state or ensure_state()
    for p in st.get("patients") or []:
        if p.get("oxx") == oxx:
            return p
    return {"oxx": oxx, "source": "", "class": "", "phase": "", "model": oxx}


def ensure_sections(pkt: dict, oxx: str, meta: dict) -> dict:
    defaults = section_defaults(oxx, meta)
    pkt.setdefault("oxx", oxx)
    if meta.get("class") and "class" not in pkt:
        pkt["class"] = meta["class"]
    if meta.get("phase") and "phase" not in pkt:
        pkt["phase"] = meta["phase"]
    for key, default in defaults.items():
        if key not in pkt:
            pkt[key] = json.loads(json.dumps(default))
        elif isinstance(default, dict) and isinstance(pkt[key], dict):
            for dk, dv in default.items():
                pkt[key].setdefault(dk, dv)
        elif key == "next" and pkt[key] is None:
            pkt[key] = []
    return pkt


def _stamp(section: dict, stamp: str) -> None:
    prev = str(section.get("_evidence") or "")
    if stamp in prev:
        return
    if not prev or prev == "UNKNOWN":
        section["_evidence"] = stamp
    else:
        section["_evidence"] = f"{stamp} + {prev}"


def apply_census(pkt: dict, census: dict) -> None:
    arch = pkt.setdefault("architecture", {})
    cfg = census.get("config") or {}
    arch["arch"] = census.get("arch") or arch.get("arch")
    arch["total_params"] = census.get("total_params")
    active = census.get("active_params_per_token")
    arch["active_params"] = active
    arch["active_params_per_token"] = active
    total = census.get("total_params")
    if total and active:
        arch["active_pct"] = round(100.0 * active / total, 1)
    arch["layers"] = cfg.get("num_hidden_layers", arch.get("layers"))
    arch["hidden_size"] = cfg.get("hidden_size", arch.get("hidden_size"))
    if census.get("is_moe"):
        arch["kind"] = "moe"
        arch["experts"] = cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts")
        arch["experts_per_tok"] = (
            cfg.get("num_experts_per_tok") or cfg.get("moe_topk") or cfg.get("n_experts_per_tok")
        )
        arch["shared_experts"] = cfg.get("n_shared_experts") or arch.get("shared_experts") or 0
    elif not arch.get("kind"):
        arch["kind"] = "dense"
    arch["census_ref"] = f"patients/{pkt['oxx']}/census.json"
    _stamp(arch, "MEASURED (census)")

    rep = pkt.setdefault("representation", {})
    rep["source_bytes"] = census.get("total_bytes")
    rep["best_stored_bpw_eq"] = census.get("stored_bpw")
    rep["stored_bpw"] = census.get("stored_bpw")
    obytes = census.get("organs_bytes") or {}
    rep["organs_bytes_GB"] = {k: round(v / 1e9, 2) for k, v in obytes.items() if v}
    if census.get("active_bytes_per_token"):
        rep["active_bytes_per_token_bf16"] = census["active_bytes_per_token"]
    _stamp(rep, "MEASURED (census)")


def apply_a3b(pkt: dict, recon: dict) -> None:
    if pkt.get("oxx") != "O005":
        return
    rm = recon.get("route_map") or {}
    routing = pkt.setdefault("routing", {})

    def _fill(dst, key, val):
        if val is not None and dst.get(key) is None:
            dst[key] = val

    if rm:
        _fill(routing, "entropy", rm.get("avg_layer_route_entropy_bits"))
        _fill(routing, "expert_frequency", {
            "most_popular_expert_share_pct": rm.get("most_popular_expert_share_pct"),
            "pct_mass_top16_experts": rm.get("pct_mass_top16_experts"),
            "never_routed_experts": rm.get("never_routed_experts"),
        })
        _fill(
            routing,
            "cold_set",
            "empty" if rm.get("never_routed_experts") == 0 else rm.get("never_routed_experts"),
        )
        _fill(routing, "hot_set", "none (near-uniform)")
        if evidence_class(routing.get("_evidence")) in (None, "UNKNOWN"):
            routing["_evidence"] = (
                "MEASURED (A3B_RECON specimen, short prompt set; A1 still required)"
            )
    ex = pkt.setdefault("execution", {})
    tps = (recon.get("A3B_baseline") or {}).get("tps_specimen")
    if tps is not None:
        ex.setdefault("tps_specimen_mlx", tps)
        # specimen is not a canonical baseline (foreign-runtime gate, bible §60)
        if evidence_class(ex.get("_evidence")) in (None, "UNKNOWN"):
            if "specimen" not in str(ex.get("_evidence") or "").lower():
                ex["_evidence"] = (
                    "UNKNOWN (canonical baseline pending; mlx specimen is not a Hawking measurement, §60)"
                )
    g = pkt.setdefault("gravity", {})
    findings = g.setdefault("architecture_specific_findings", [])
    for key, val in (recon.get("CLASSIFICATION") or {}).items():
        note = f"{key}: {val}"
        if note not in findings:
            findings.append(note)
    if findings and evidence_class(g.get("_evidence")) in (None, "UNKNOWN"):
        g["_evidence"] = "INFERRED (A3B_RECON CLASSIFICATION)"


def apply_transfer(pkt: dict) -> None:
    if not TRANSFER.is_file():
        return
    try:
        matrix = read_json(TRANSFER)
    except (OSError, json.JSONDecodeError):
        return
    oxx = pkt.get("oxx")
    tr = pkt.setdefault("transfer", {})
    inherited, unchanged, retuned, failed, harmful = [], [], [], [], []
    for row in matrix.get("rows") or []:
        rid = row.get("rule")
        cell = (row.get("cells") or {}).get(oxx)
        if not rid or not cell or cell == "NOT_TESTED":
            continue
        inherited.append(rid)
        if cell == "TRANSFERRED_UNCHANGED":
            unchanged.append(rid)
        elif cell in ("TRANSFERRED_RETUNED", "RETUNED"):
            retuned.append(rid)
        elif cell == "FAILED":
            failed.append(rid)
        elif cell == "HARMFUL":
            harmful.append(rid)
    if inherited:
        tr["inherited_rules"] = inherited
        tr["unchanged"] = unchanged
        tr["retuned"] = retuned
        tr["failed"] = failed
        tr["harmful"] = harmful
        if evidence_class(tr.get("_evidence")) in (None, "UNKNOWN"):
            tr["_evidence"] = "INFERRED (TRANSFER_MATRIX.json)"


def apply_receipts(pkt: dict, oxx: str) -> None:
    if not RECEIPT_DIR.is_dir():
        return
    findings = []
    for path in sorted(RECEIPT_DIR.glob("*.json")):
        try:
            rec = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("oxx") != oxx or rec.get("verdict") != "ACCEPTED":
            continue
        body = rec.get("outputs") or {}
        if isinstance(body, dict) and body.get("finding"):
            findings.append(str(body["finding"]))
        ev = rec.get("evidence")
        payload = body.get("structured") if isinstance(body, dict) else None
        if isinstance(payload, dict):
            if "routing" in payload and isinstance(payload["routing"], dict):
                pkt.setdefault("routing", {}).update(payload["routing"])
            if "gravity" in payload and isinstance(payload["gravity"], dict):
                g = pkt.setdefault("gravity", {})
                for key in ("tried_mechanisms", "wins", "kills"):
                    extra = payload["gravity"].get(key) or []
                    bucket = g.setdefault(key, [])
                    for item in extra:
                        if item not in bucket:
                            bucket.append(item)
        if ev and ev not in (pkt.get("harvest_evidence") or []):
            pkt.setdefault("harvest_evidence", []).append(ev)
    nxt = pkt.get("next")
    if isinstance(nxt, list):
        for f in findings:
            line = f"harvest: {f}"
            if line not in nxt:
                nxt.append(line)


def seal_packet(pkt: dict) -> None:
    """Reuse doctor_seal.seal for every patient seal (structural only)."""
    doctor = pkt.get("doctor") or {}
    candidate = {
        "tabula_drift": (pkt.get("tabula") or {}).get("drift_monitor") or doctor.get("tabula_drift"),
        "observed_controls": doctor.get("controls") or doctor.get("observed_controls"),
        "stated_test_width": doctor.get("stated_test_width"),
        "known_blind_spots": doctor.get("blind_spots") or doctor.get("known_blind_spots"),
    }
    verdict, reasons = doctor_seal.seal(candidate)
    pkt["receipt_seal"] = {
        "verdict": verdict,
        "reasons": reasons,
        "_evidence": "DERIVED (doctor_seal.seal structural)",
    }


def validate_packet(pkt: dict) -> list[str]:
    errs = []
    if not pkt.get("oxx"):
        errs.append("missing oxx")
    for sec in PACKET_SECTIONS:
        if sec not in pkt:
            errs.append(f"missing section {sec}")
            continue
        val = pkt[sec]
        if isinstance(val, dict):
            ev = evidence_class(val.get("_evidence"))
            if ev is None:
                errs.append(f"{sec} missing §18 _evidence")
        elif sec == "next" and not isinstance(val, (list, dict, str)):
            errs.append("next must be list/dict/str")
    return errs


def assemble_packet(oxx: str, state: dict | None = None) -> dict:
    oxx = norm_oxx(oxx)
    meta = patient_meta(oxx, state)
    pkt = load_packet(oxx) or {"oxx": oxx}
    pkt = ensure_sections(pkt, oxx, meta)
    cp = census_path(oxx)
    if cp.is_file():
        apply_census(pkt, read_json(cp))
    if A3B_RECON.is_file():
        try:
            apply_a3b(pkt, read_json(A3B_RECON))
        except (OSError, json.JSONDecodeError):
            pass
    apply_transfer(pkt)
    apply_receipts(pkt, oxx)
    if not pkt.get("next"):
        st = state or ensure_state()
        pkt["next"] = [
            f"{w['id']} {w['title']}"
            for w in st.get("work") or []
            if w.get("oxx") == oxx and w.get("status") == "READY"
        ]
    seal_packet(pkt)
    return pkt


def write_packet(oxx: str, state: dict | None = None) -> Path:
    oxx = norm_oxx(oxx)
    pkt = assemble_packet(oxx, state)
    errs = validate_packet(pkt)
    if errs:
        raise SystemExit(f"packet {oxx} invalid: {errs}")
    dest = packet_path(oxx)
    write_json(dest, pkt)
    return dest


# ---------------------------------------------------------------------------
# harvest (§12)
# ---------------------------------------------------------------------------

def classify_evidence(text: str, structured=None) -> str:
    if isinstance(structured, dict):
        ev = evidence_class(structured.get("evidence") or structured.get("_evidence"))
        if ev:
            return ev
    hits = [ev for ev in EVIDENCE if re.search(rf"\b{ev}\b", text)]
    if not hits:
        return "UNKNOWN"
    # strongest first (EVIDENCE is already strongest-first)
    for ev in EVIDENCE:
        if ev in hits:
            return ev
    return "UNKNOWN"


def parse_report(text: str) -> dict:
    structured = None
    m = JSON_FENCE_RE.search(text)
    if m:
        try:
            structured = json.loads(m.group(1))
        except json.JSONDecodeError:
            structured = None
    completion = None
    cm = COMPLETION_RE.search(text)
    if cm:
        completion = cm.group(1).strip()[:2000]
    has_result = bool(RESULT_RE.search(text))
    ok = structured is not None or completion is not None or has_result
    oxx = None
    if isinstance(structured, dict) and structured.get("oxx"):
        try:
            oxx = norm_oxx(str(structured["oxx"]))
        except (TypeError, ValueError):
            oxx = None
    if oxx is None:
        hm = OXX_RE.search(text)
        if hm:
            oxx = f"O{hm.group(1)}"
    return {
        "structured": structured,
        "completion": completion,
        "ok": ok,
        "oxx": oxx,
        "evidence": classify_evidence(text, structured),
    }


def oxx_from_task(name: str, parsed: dict) -> str | None:
    if parsed.get("oxx"):
        return parsed["oxx"]
    m = SLUG_OXX_RE.search(name)
    return f"O{m.group(1)}" if m else None


def should_escalate(text: str, evidence: str) -> tuple[str, str] | None:
    if evidence == "REFUTED":
        return "CONTRADICTION", "evidence classified REFUTED"
    if re.search(r"two (high-quality )?receipts disagree", text, re.I):
        return "CONTRADICTION", "receipts disagree"
    if re.search(r"\bANOMALY\b|\bESCALATE(?: TO OPUS)?\b", text):
        return "ANOMALY", "explicit nomination"
    if re.search(r"\bHARMFUL\b.*\btransfer\b", text, re.I):
        return "MAJOR FALSE WIN", "harmful transfer"
    return None


def _task_status(task_dir: Path) -> str:
    p = task_dir / "status"
    if not p.is_file():
        return "unknown"
    return p.read_text(errors="replace").strip().splitlines()[0].strip().lower()


def harvest(*, tasks_root: Path | None = None, receipt_dir: Path | None = None,
            escalate_path: Path | None = None, state: dict | None = None) -> list[dict]:
    """Scan completed odyssey-* grok lanes. Reject reports with no structured result."""
    root = Path(tasks_root) if tasks_root else GROK_TASKS
    out_dir = Path(receipt_dir) if receipt_dir else RECEIPT_DIR
    esc_path = Path(escalate_path) if escalate_path else ESCALATIONS
    st = state if state is not None else ensure_state()
    already = set(st.get("harvested") or [])
    rows = []
    if not root.is_dir():
        return rows
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("odyssey-")):
        report = d / "grok-report.md"
        stat = _task_status(d)
        if stat in {"running", "queued", "starting"}:
            continue
        rec = {
            "schema": "hawking.odyssey.harvest.v1",
            "task": d.name,
            "slug": re.sub(r"-\d{8}-\d{6}$", "", d.name),
            "task_status": stat,
            "verdict": "REJECTED",
            "reason": "",
            "evidence": "UNKNOWN",
            "oxx": None,
            "patient_hash": None,
            "representation_hash": None,
            "nx_hash": None,
            "machine_state": None,
            "command": "harvest",
            "inputs": {"task_dir": str(d)},
            "outputs": {},
            "controls": [],
            "assumptions": ["only odyssey-* slugs; running lanes skipped"],
            "reopen_if": "structured result later appears in grok-report.md",
            "_evidence": "DERIVED (harvester)",
        }
        if not report.is_file():
            rec["reason"] = "malformed: no grok-report.md"
        else:
            try:
                text = report.read_text(errors="replace")
            except OSError as exc:
                rec["reason"] = f"malformed: unreadable report ({exc})"
                text = ""
            if text:
                parsed = parse_report(text)
                rec["oxx"] = oxx_from_task(d.name, parsed)
                rec["evidence"] = parsed["evidence"]
                rec["_evidence"] = parsed["evidence"]
                if not parsed["ok"]:
                    rec["reason"] = "malformed: no structured result"
                else:
                    rec["verdict"] = "ACCEPTED"
                    rec["reason"] = "ok"
                    rec["outputs"] = {
                        "finding": (parsed.get("completion") or "")[:400],
                        "structured": parsed.get("structured"),
                    }
                    trig = should_escalate(text, parsed["evidence"])
                    if trig:
                        rec["escalation"] = {"trigger": trig[0], "note": trig[1]}
                        esc_path.parent.mkdir(parents=True, exist_ok=True)
                        with esc_path.open("a") as fh:
                            fh.write(json.dumps({
                                "task": d.name,
                                "oxx": rec["oxx"],
                                "trigger": trig[0],
                                "note": trig[1],
                                "evidence": parsed["evidence"],
                                "_evidence": "INFERRED (harvester nomination, bible §9/§12)",
                            }) + "\n")
        dest = out_dir / f"{d.name}.json"
        write_json(dest, rec)
        rows.append({"task": d.name, "verdict": rec["verdict"],
                     "reason": rec["reason"], "oxx": rec["oxx"],
                     "receipt": str(dest)})
        if rec["verdict"] == "ACCEPTED" and rec["oxx"] and receipt_dir is None:
            # only mutate real packets when writing to the real receipt dir
            try:
                write_packet(rec["oxx"], st)
            except SystemExit:
                pass
        if d.name not in already:
            already.add(d.name)
    st["harvested"] = sorted(already)
    if receipt_dir is None and escalate_path is None and tasks_root is None:
        save_state(st)
    return rows


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def live_odyssey_lanes() -> list[str]:
    if not GROK_TASKS.is_dir():
        return []
    live = []
    for d in GROK_TASKS.iterdir():
        if d.is_dir() and d.name.startswith("odyssey-") and _task_status(d) == "running":
            live.append(d.name)
    return sorted(live)


def frontier_counts() -> dict:
    out = {
        "universal_rules": 0,
        "architecture_rules": 0,
        "new_negatives": 0,
        "transfer_score": "0/0",
        "_evidence": "INFERRED",
    }
    rules = []
    if RULEBASE.is_file():
        try:
            rules = read_json(RULEBASE).get("rules") or []
        except (OSError, json.JSONDecodeError):
            rules = []
    if NEGATIVE.is_file():
        try:
            out["new_negatives"] = len(read_json(NEGATIVE).get("entries") or [])
        except (OSError, json.JSONDecodeError):
            pass
    tested = total = 0
    arch_rules = set()
    families_ok = {}
    if TRANSFER.is_file():
        try:
            matrix = read_json(TRANSFER)
        except (OSError, json.JSONDecodeError):
            matrix = {}
        for row in matrix.get("rows") or []:
            rid = row.get("rule")
            cells = row.get("cells") or {}
            fams = set()
            for oxx, cell in cells.items():
                total += 1
                if cell != "NOT_TESTED":
                    tested += 1
                if cell == "ARCHITECTURE_SPECIFIC":
                    arch_rules.add(rid)
                if cell == "TRANSFERRED_UNCHANGED":
                    fams.add("moe" if oxx in MOE_PATIENTS else "other")
            families_ok[rid] = fams
    out["architecture_rules"] = len(arch_rules) or sum(
        1 for r in rules if "moe" in json.dumps(r).lower()
    )
    out["universal_rules"] = sum(1 for fams in families_ok.values() if len(fams) >= 2)
    out["transfer_score"] = f"{tested}/{total}" if total else "0/0"
    return out


def ready_work(state: dict) -> list[dict]:
    blocked = {p["oxx"] for p in state.get("patients") or [] if p.get("state") == "BLOCKED"}
    out = []
    for w in state.get("work") or []:
        if w.get("status") != "READY":
            continue
        if w.get("oxx") in blocked and w.get("kind") != "acquisition":
            continue
        if w.get("id") == "A6":
            continue
        out.append(w)
    out.sort(key=value, reverse=True)
    return out


def cmd_status() -> int:
    state = ensure_state()
    patients = {p["oxx"]: p for p in state.get("patients") or []}
    on_disk = [p["oxx"] for p in state["patients"] if p.get("on_disk") or
               str(p.get("ledger", "")).lower().startswith("on-disk")]
    blocked = [p["oxx"] for p in state["patients"] if p.get("state") == "BLOCKED"]
    queued = [p["oxx"] for p in state["patients"]
              if str(p.get("ledger", "")).lower().startswith("queued")]
    # PATIENT of record = already-in-motion teacher (O005), not the cheapest next action.
    # NEXT is where §22 ranking lives.
    ranked = ready_work(state)
    if "O005" in on_disk:
        current = "O005"
    elif on_disk:
        current = on_disk[0]
    elif ranked:
        current = ranked[0]["oxx"]
    else:
        current = "O005"
    meta = patients.get(current) or patient_meta(current, state)
    pkt = load_packet(current) or assemble_packet(current, state)
    doctor = pkt.get("doctor") or {}
    doctor_s = doctor.get("fast_doctor_seal_ref") or "UNKNOWN"
    if doctor_s in (None, ""):
        doctor_s = evidence_class(doctor.get("_evidence")) or "UNKNOWN"
    rep = pkt.get("representation") or {}
    bpw = rep.get("stored_bpw") or rep.get("best_stored_bpw_eq")
    bpw_ev = evidence_class(rep.get("_evidence")) or "UNKNOWN"
    active = rep.get("active_bytes_per_token_bf16") or rep.get("active_learned_bytes_per_token")
    ex = pkt.get("execution") or {}
    tps = ex.get("baseline_tps") or ex.get("tps")
    tps_ev = evidence_class(ex.get("_evidence")) or "UNKNOWN"
    finding = (
        (rep.get("lever") if isinstance(rep.get("lever"), str) else None)
        or (meta.get("class") or "")
    )
    fc = frontier_counts()
    lanes = live_odyssey_lanes()
    esc_n = 0
    if ESCALATIONS.is_file():
        esc_n = sum(1 for line in ESCALATIONS.read_text().splitlines() if line.strip())
    nxt = [w for w in ranked[:3]]
    next_on_disk = [x for x in on_disk if x != current]
    print("HAWKING ODYSSEY-I")
    print()
    print("QUEUE:")
    print(f"    on-disk: {','.join(on_disk) or '—'}")
    print(f"    BLOCKED-auth: {','.join(blocked) or '—'}")
    print(f"    queued: {','.join(queued) or '—'}")
    print()
    print("PATIENT:")
    print(f"    {current}")
    print(f"    phase: {pkt.get('phase') or meta.get('phase') or '—'}")
    print(f"    source: {(pkt.get('identity') or {}).get('source_repo') or meta.get('source') or '—'}")
    print(f"    Doctor: {doctor_s}")
    print(f"    stored BPW: {bpw if bpw is not None else 'UNKNOWN'}"
          + (f" {bpw_ev}" if bpw is not None else ""))
    print(f"    active bytes/token: {active if active is not None else 'UNKNOWN'}"
          + (f" {bpw_ev}" if active is not None else ""))
    tps_s = "UNKNOWN" if tps in (None, "", "UNKNOWN") or (
        isinstance(tps, str) and tps.startswith("UNKNOWN")
    ) else tps
    print(f"    TPS: {tps_s}" + (f" {tps_ev}" if tps_s != "UNKNOWN" else ""))
    print(f"    top finding: {finding}")
    print()
    print("NEXT PATIENT:")
    acq = ", ".join(next_on_disk) if next_on_disk else "—"
    print(f"    acquisition: {acq}; queued: {','.join(queued) or '—'}")
    print(f"    static recon: {'/'.join(blocked) or '—'} BLOCKED-auth")
    print()
    print("COMPILER:")
    print(f"    universal rules: {fc['universal_rules']} {fc['_evidence']}")
    print(f"    architecture rules: {fc['architecture_rules']} {fc['_evidence']}")
    print(f"    new negatives: {fc['new_negatives']} {fc['_evidence']}")
    print(f"    transfer score: {fc['transfer_score']} {fc['_evidence']}")
    print()
    print("RESEARCH:")
    print(f"    Grok lanes: {len(lanes)} MEASURED (live odyssey-*)")
    print("    deterministic jobs: 0")
    print(f"    GPU owner: {(state.get('metrics') or {}).get('gpu_owner', 'none')}")
    print(f"    Opus escalations: {esc_n}")
    print()
    print("NEXT:")
    if not nxt:
        print("    (empty READY queue)")
    for i, w in enumerate(nxt, 1):
        print(f"    {i}. {w['id']} {w['oxx']} {w['title']}  "
              f"proxy={value(w):.1f} HYPOTHESIS")
    return 0


def cmd_queue() -> int:
    state = ensure_state()
    print(f"{'Oxx':<6}{'state':<12}{'phase':<10}{'ledger':<28}model")
    for p in state.get("patients") or []:
        st = p.get("state") or "?"
        if st not in STATES:
            continue
        print(f"{p['oxx']:<6}{st:<12}{(p.get('phase') or '—'):<10}"
              f"{(p.get('ledger') or '—'):<28}{p.get('model') or ''}")
    return 0


def cmd_value() -> int:
    state = ensure_state()
    ranked = ready_work(state)
    print("info-value proxy = info / (wall+gpu+opus)   HYPOTHESIS (§22, ordering only)")
    print(f"{'proxy':>6}  {'id':<10}{'oxx':<6}{'kind':<32}title")
    for w in ranked:
        print(f"{value(w):6.1f}  {w['id']:<10}{w['oxx']:<6}{(w.get('kind') or ''):<32}{w['title']}")
    return 0


def cmd_harvest() -> int:
    rows = harvest()
    if not rows:
        print("harvest: no completed odyssey-* tasks")
        return 0
    acc = sum(1 for r in rows if r["verdict"] == "ACCEPTED")
    rej = sum(1 for r in rows if r["verdict"] == "REJECTED")
    print(f"harvest: {acc} accepted, {rej} rejected")
    for r in rows:
        print(f"  {r['verdict']:<8} {r['task']:<42} {r.get('oxx') or '—':<5} {r['reason']}")
    return 0


def cmd_packet(oxx: str) -> int:
    dest = write_packet(oxx)
    print(f"wrote {dest.relative_to(REPO)}  valid")
    return 0


def cmd_admit(slug: str, est_gib: float) -> int:
    """Call worker_gate before any model-loading worker. Abort on REFUSE."""
    try:
        obs = worker_gate.observe()
        g = worker_gate.gate(obs)
    except Exception as exc:
        print(f"REFUSE  slug={slug} est_gib={est_gib}  worker_gate failed: {exc}")
        return 1
    snap = machine_snapshot()
    disk = snap.get("disk_free_gib")
    decision = "GO" if g.get("decision") == "PERMIT" else "REFUSE"
    notes = [g.get("note") or ""]
    if disk is not None and disk < DISK_FLOOR_GIB:
        decision = "REFUSE"
        notes.append(f"disk {disk} GiB below floor {DISK_FLOOR_GIB}")
        reclaim_if_tight(snap)
    elif disk is not None and (disk - est_gib) < DISK_FLOOR_GIB:
        decision = "REFUSE"
        notes.append(
            f"est {est_gib} GiB would leave {disk - est_gib:.1f} GiB (< {DISK_FLOOR_GIB} floor)"
        )
    note = "; ".join(n for n in notes if n)
    print(f"{decision}  slug={slug} est_gib={est_gib}  {note}")
    print(f"  gate={g.get('decision')} wired={g.get('current_wired_gb')} "
          f"headroom={g.get('projected_headroom_gb')}  _evidence=MEASURED (worker_gate)")
    return 0 if decision == "GO" else 1


def _self_check() -> int:
    # 1. state round-trip
    st = ensure_state()
    save_state(st)
    st2 = read_json(STATE)
    assert st2.get("schema") == SCHEMA, st2.get("schema")
    assert len(st2.get("patients") or []) >= 14, len(st2.get("patients") or [])
    oxxs = [p["oxx"] for p in st2["patients"]]
    assert oxxs == [f"O{i:03d}" for i in range(14)], oxxs
    by = {p["oxx"]: p for p in st2["patients"]}
    assert by["O000"]["state"] == "BLOCKED" and "BLOCKED-auth" in by["O000"]["ledger"]
    assert by["O002"]["state"] == "BLOCKED"
    assert by["O004"]["state"] == "BLOCKED"
    assert by["O001"]["on_disk"] and by["O001"]["state"] == "READY"
    assert by["O005"]["on_disk"] and by["O005"]["state"] == "READY"
    assert str(by["O003"]["ledger"]).lower().startswith("queued")
    save_state(st2)
    assert read_json(STATE) == st2, "state did not round-trip"

    # 2. value ranking: cheap architecture-first > expensive acquisition
    cheap = {"info": 8, "wall_cost": 1, "gpu_cost": 0, "opus_cost": 0}
    dear = {"info": 3, "wall_cost": 8, "gpu_cost": 0, "opus_cost": 0}
    assert value(cheap) > value(dear), (value(cheap), value(dear))
    assert value({"info": 1, "wall_cost": 0, "gpu_cost": 0, "opus_cost": 0}) == 10.0

    # 3. status renders the seed queue
    from io import StringIO
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = cmd_status()
    finally:
        sys.stdout = old
    assert rc == 0
    text = buf.getvalue()
    assert "HAWKING ODYSSEY-I" in text, text[:200]
    assert "O005" in text and "O001" in text
    assert "on-disk" in text
    assert "BLOCKED-auth" in text
    assert "O000" in text and "O002" in text and "O004" in text
    assert "queued" in text

    # 4. packet builder validates against schema (refresh seed + stub)
    for oxx in ("O005", "O001"):
        pkt = assemble_packet(oxx, st2)
        errs = validate_packet(pkt)
        assert not errs, (oxx, errs)
        dest = write_packet(oxx, st2)
        again = read_json(dest)
        assert validate_packet(again) == []
        # idempotent: second assemble matches
        assert assemble_packet(oxx, st2) == again
    stub = assemble_packet("O003", st2)
    assert validate_packet(stub) == []
    assert stub["architecture"]["kind"] == "moe"

    schema = read_json(SCHEMA_PATH)
    for sec in schema.get("fields") or {}:
        assert sec in assemble_packet("O005", st2), sec

    # 5. harvest tolerates a malformed task dir
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tasks, recs, esc = td / "tasks", td / "recs", td / "esc.jsonl"
        mal = tasks / "odyssey-malformed-xyz"
        mal.mkdir(parents=True)
        (mal / "status").write_text("done\n")
        (mal / "grok-report.md").write_text("just chatting, no result\n")
        empty = tasks / "odyssey-empty-xyz"
        empty.mkdir()
        (empty / "status").write_text("done\n")
        running = tasks / "odyssey-running-xyz"
        running.mkdir()
        (running / "status").write_text("running\n")
        ignore = tasks / "not-odyssey-xyz"
        ignore.mkdir()
        (ignore / "grok-report.md").write_text("**Completion report**\n\nok\n")
        good = tasks / "odyssey-o005-fixture-xyz"
        good.mkdir()
        (good / "status").write_text("done\n")
        (good / "grok-report.md").write_text(
            "**Completion report**\n\n"
            "```json\n"
            + json.dumps({"oxx": "O005", "finding": "fixture", "evidence": "HYPOTHESIS"})
            + "\n```\n"
        )
        rows = harvest(tasks_root=tasks, receipt_dir=recs, escalate_path=esc, state=dict(st2))
        by_task = {r["task"]: r for r in rows}
        assert "odyssey-running-xyz" not in by_task
        assert by_task["odyssey-malformed-xyz"]["verdict"] == "REJECTED"
        assert by_task["odyssey-empty-xyz"]["verdict"] == "REJECTED"
        assert by_task["odyssey-o005-fixture-xyz"]["verdict"] == "ACCEPTED"
        assert by_task["odyssey-o005-fixture-xyz"]["oxx"] == "O005"
        assert (recs / "odyssey-malformed-xyz.json").is_file()

    # 6. governors: worker_gate refuses injected pressure; doctor_seal refuses empty
    obs = {
        "total_gb": 100.0, "wired_gb": 4.63, "free_gb": 50.0, "inactive_gb": 10.0,
        "compressed_gb": 0.05, "swap_used_mb": 512.0, "workers_resident": 0,
        "worker_rss_total_gb": 0.0,
    }
    g = worker_gate.gate(obs)
    assert g["decision"] == "REFUSE", g
    v, _ = doctor_seal.seal({})
    assert v == "REFUSED", v

    # 7. frontier files exist, parse, non-empty, labelled
    for path, key in ((RULEBASE, "rules"), (TRANSFER, "rows"), (NEGATIVE, "entries")):
        assert path.is_file(), path
        doc = read_json(path)
        assert evidence_class(doc.get("_evidence")), path
        assert doc.get(key), path
    cells0 = (read_json(TRANSFER)["rows"][0].get("cells") or {})
    for i in range(14):
        assert f"O{i:03d}" in cells0, f"missing O{i:03d}"

    print("self-check ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="odyssey_ctl")
    ap.add_argument("--self-check", action="store_true",
                    help="assert state/status/packet/harvest; no network, no model")
    sp = ap.add_subparsers(dest="cmd")
    sp.add_parser("status")
    sp.add_parser("queue")
    sp.add_parser("value")
    sp.add_parser("harvest")
    p_pkt = sp.add_parser("packet")
    p_pkt.add_argument("oxx")
    p_ad = sp.add_parser("admit")
    p_ad.add_argument("slug")
    p_ad.add_argument("est_gib", type=float)
    args = ap.parse_args(argv)
    if args.self_check or args.cmd in {"self-check", "selfcheck"}:
        return _self_check()
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "queue":
        return cmd_queue()
    if args.cmd == "value":
        return cmd_value()
    if args.cmd == "harvest":
        return cmd_harvest()
    if args.cmd == "packet":
        return cmd_packet(args.oxx)
    if args.cmd == "admit":
        return cmd_admit(args.slug, args.est_gib)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
