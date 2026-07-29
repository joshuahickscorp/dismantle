#!/usr/bin/env python3.12
"""One operator registry for campaign composition.

Campaigns are typed specs composed of operators. Operators do not own a CLI,
download loop, status writer, hash chain, checkpoint, lease, or report
formatter — those live in the engine. An operator is science, adaptation, or a
small numerical authority that the runtime may invoke by key.

Classification classes (Track V behaviour contract):

* ``operator`` — pure work unit (pack, encode, forward, assemble, …)
* ``spec`` — declarative campaign data (not a Python module body)
* ``dataset_adapter`` — corpus / fixture / stream adapters
* ``eval_rule`` — parity, gauntlet, terminal proofs, authority gates
* ``fixture`` — synthetic / real fixture builders
* ``numerical_authority`` — metrics, ledgers, null models, kernels
* ``UNCLASSIFIED`` — does not fit the six; named in the registry with why

Path-sealed modules (sha256-bound to ``tools/condense/<name>.py``) stay at
their historical import paths. The registry is the authority for *role*, not
filesystem layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

# Handler type matches runtime.Handler without importing runtime (cycle-safe).
Handler = Callable[..., dict[str, Any]]


class OperatorClass(str, Enum):
    OPERATOR = "operator"
    SPEC = "spec"
    DATASET_ADAPTER = "dataset_adapter"
    EVAL_RULE = "eval_rule"
    FIXTURE = "fixture"
    NUMERICAL_AUTHORITY = "numerical_authority"
    UNCLASSIFIED = "UNCLASSIFIED"


# Path to tools/condense (parent of engine/).
CONDENSE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OperatorRecord:
    """One registered operator / residual module."""

    module: str
    class_: OperatorClass
    loc: int
    why: str
    path: str = ""
    # Engine handler key(s) this module may satisfy when wired.
    handler_keys: tuple[str, ...] = ()
    # True when sealed receipts bind the path or content hash of this file.
    path_sealed: bool = False
    # True when the body is live science / control-plane domain, not architecture.
    science: bool = False

    @property
    def class_name(self) -> str:
        return self.class_.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "class": self.class_.value,
            "loc": self.loc,
            "why": self.why,
            "path": self.path or f"tools/condense/{self.module}.py",
            "handler_keys": list(self.handler_keys),
            "path_sealed": self.path_sealed,
            "science": self.science,
        }


def _loc(name: str) -> int:
    path = CONDENSE_ROOT / f"{name}.py"
    if not path.is_file():
        return 0
    return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())


def _r(
    module: str,
    class_: OperatorClass,
    why: str,
    *,
    handler_keys: tuple[str, ...] = (),
    path_sealed: bool = False,
    science: bool = False,
    loc: int | None = None,
) -> OperatorRecord:
    return OperatorRecord(
        module=module,
        class_=class_,
        loc=loc if loc is not None else _loc(module),
        why=why,
        path=f"tools/condense/{module}.py",
        handler_keys=handler_keys,
        path_sealed=path_sealed,
        science=science,
    )


# ---------------------------------------------------------------------------
# Canonical classification of every top-level tools/condense module.
# This is the Track V behaviour contract: every module is named and classed.
# ---------------------------------------------------------------------------

IRREDUCIBLE_MODULES: tuple[OperatorRecord, ...] = (
    # --- shared path authority ------------------------------------------------
    _r(
        "glm52_common",
        OperatorClass.NUMERICAL_AUTHORITY,
        "Single path authority resolve_artifact + canonical seal helpers; "
        "path-sealed into contracts and corpus receipts.",
        handler_keys=("artifact.resolve",),
        path_sealed=True,
        science=False,
    ),
    # --- residual controller (architecture residue) ---------------------------
    _r(
        "glm52_state",
        OperatorClass.UNCLASSIFIED,
        "Partially decomposed residual controller. Lease half absorbed into "
        "engine.lease.SingletonLease (TOCTOU-hardened flock, parent-chain "
        "O_NOFOLLOW open, post-lock revalidation). Remaining body is "
        "HashChainLog + WindowLedger + TrustedArtifactStore + Controller + "
        "GLM52 evidence validators — dual-log claim-bound transitions under "
        "exclusive lease with official shard/coverage contracts. Thin "
        "SingletonLease subclass maps StateError; dies when Controller is "
        "absorbed.",
        handler_keys=("state.controller",),
        path_sealed=True,
        science=False,  # residual architecture + domain, not pack/parity/gravity floor
    ),
    # --- live GLM52 readers (must keep working) -------------------------------
    _r(
        "glm52_parity",
        OperatorClass.EVAL_RULE,
        "Adapter/twin/reference parity instrument; sealed parity surface.",
        handler_keys=("measure.parity",),
        path_sealed=True,
        science=True,
    ),
    _r(
        "glm52_contract",
        OperatorClass.EVAL_RULE,
        "Immutable source contract and header-derived ledgers; live reader.",
        handler_keys=("precheck.contract",),
        path_sealed=True,
        science=True,
    ),
    _r(
        "glm52_source_fetch",
        OperatorClass.DATASET_ADAPTER,
        "BF16 source streamer with sealed schedule and manifest verification.",
        handler_keys=("fetch.source",),
        path_sealed=False,
        science=True,
    ),
    _r(
        "glm52_teacher_capture",
        OperatorClass.OPERATOR,
        "Teacher-evidence capture on the BF16 stream before eviction.",
        handler_keys=("measure.teacher_capture",),
        path_sealed=False,
        science=True,
    ),
    _r(
        "glm52_xet_autotune",
        OperatorClass.EVAL_RULE,
        "Offline planner and fail-closed authority gate for Xet autotuning "
        "(adjacent to pack science; not in the pack/parity/gravity floor).",
        handler_keys=("plan.xet",),
        path_sealed=True,
        science=False,
    ),
    # --- pack / activation science --------------------------------------------
    _r(
        "glm52_activation_aware_pack",
        OperatorClass.OPERATOR,
        "Activation-aware packing program v1 (real-activation pilot order).",
        handler_keys=("pack.activation_v1",),
        science=True,
    ),
    _r(
        "glm52_activation_aware_pack_v2",
        OperatorClass.OPERATOR,
        "Activation-aware pack v2 — feasibility + fake-data codec (VARIANT of v1).",
        handler_keys=("pack.activation_v2",),
        science=True,
    ),
    _r(
        "glm52_pack",
        OperatorClass.OPERATOR,
        "Serialize tensors into physically-exact sub-bit compact shards.",
        handler_keys=("pack.stream",),
        science=True,
    ),
    _r(
        "glm52_adapter",
        OperatorClass.DATASET_ADAPTER,
        "Fail-closed checkpoint adapter and bounded safetensors reader.",
        handler_keys=("adapt.checkpoint",),
        path_sealed=True,
        science=True,
    ),
    _r(
        "glm52_assemble",
        OperatorClass.OPERATOR,
        "Assemble packed shards into one verified local model, or say why not.",
        handler_keys=("assemble.model",),
        science=True,
    ),
    _r(
        "glm52_capture_program",
        OperatorClass.DATASET_ADAPTER,
        "Natural teacher-capture program: real text, disjoint splits, domains.",
        handler_keys=("capture.program",),
        science=True,
    ),
    _r(
        "glm52_corpus",
        OperatorClass.DATASET_ADAPTER,
        "Offline quality-corpus integrity contract; no model/network code.",
        handler_keys=("corpus.verify",),
        path_sealed=True,
        science=False,  # quality corpus integrity; adjacent, not pack codec
    ),
    _r(
        "glm52_evidence_auth",
        OperatorClass.EVAL_RULE,
        "Keychain-backed producer authentication for campaign evidence.",
        handler_keys=("auth.evidence",),
        science=False,
    ),
    _r(
        "glm52_functional_gauntlet",
        OperatorClass.EVAL_RULE,
        "FS0–FS6 functional-escape survival suite across layers/documents.",
        handler_keys=("eval.gauntlet",),
        science=True,
    ),
    _r(
        "glm52_grounding",
        OperatorClass.EVAL_RULE,
        "Fail-closed read-only grounding observations (no write/network).",
        handler_keys=("eval.grounding",),
        science=True,
    ),
    _r(
        "glm52_grounding_auth",
        OperatorClass.EVAL_RULE,
        "Independent Keychain credential for filesystem observations.",
        handler_keys=("auth.grounding",),
        science=False,
    ),
    _r(
        "glm52_moe_student",
        OperatorClass.NUMERICAL_AUTHORITY,
        "Dense random-feature MoE student with ridge fit — function not weights.",
        handler_keys=("student.moe",),
        science=True,
    ),
    _r(
        "glm52_reference",
        OperatorClass.NUMERICAL_AUTHORITY,
        "Inspectable NumPy reference forward for main + physical MTP (oracle).",
        handler_keys=("oracle.reference",),
        path_sealed=True,
        science=True,
    ),
    _r(
        "glm52_shard_probe",
        OperatorClass.OPERATOR,
        "One-pass weight evidence capture for a resident BF16 source shard.",
        handler_keys=("probe.shard",),
        science=True,
    ),
    _r(
        "glm52_synthetic",
        OperatorClass.FIXTURE,
        "Deterministic architecture-preserving safetensors fixture builder.",
        handler_keys=("fixture.synthetic",),
        path_sealed=True,
        science=True,
    ),
    _r(
        "glm52_telegram",
        OperatorClass.OPERATOR,
        "Secure Telegram credentials and delivery for campaign alerts.",
        handler_keys=("notify.telegram",),
        science=False,  # ops surface, not pack science
    ),
    _r(
        "glm52_terminal_proofs",
        OperatorClass.EVAL_RULE,
        "Pure semantic proofs for offline-ready stop conditions.",
        handler_keys=("eval.terminal_proofs",),
        path_sealed=True,
        science=True,
    ),
    _r(
        "glm52_xet_live",
        OperatorClass.OPERATOR,
        "Authority-gated body-file-free live Xet trials (VARIANT of autotune).",
        handler_keys=("exec.xet_live",),
        path_sealed=True,
        science=False,
    ),
    # --- GPT-OSS / doctor (related lab science; outside narrow floor) ---------
    _r(
        "doctor_v5_gptoss_mxfp4",
        OperatorClass.OPERATOR,
        "Bounded-memory GPT-OSS MXFP4 inventory and staging primitives.",
        handler_keys=("doctor.mxfp4",),
        science=False,
    ),
    _r(
        "gptoss_block",
        OperatorClass.NUMERICAL_AUTHORITY,
        "Bounded single-block GPT-OSS-120B forward producing real MoE input.",
        handler_keys=("forward.gptoss_block",),
        science=False,
    ),
    _r(
        "gptoss_moe_runtime",
        OperatorClass.OPERATOR,
        "Per-expert STR2 loader + CPU-reference MoE runtime.",
        handler_keys=("forward.gptoss_moe",),
        science=False,
    ),
    _r(
        "gptoss_real_forward",
        OperatorClass.OPERATOR,
        "Full-model GPT-OSS-120B bounded-streaming parity-correct forward.",
        handler_keys=("forward.gptoss_real",),
        science=False,
    ),
    _r(
        "gptoss_subbit_packer",
        OperatorClass.OPERATOR,
        "Sub-1-bit deployable packer on Metal/MPS.",
        handler_keys=("pack.gptoss_subbit",),
        science=True,  # sub-bit pack science (Gravity mechanism)
    ),
    # --- gravity science ------------------------------------------------------
    _r(
        "gravity_bench_lab",
        OperatorClass.EVAL_RULE,
        "Matched-benchmark harness — every speed claim must pass through here.",
        handler_keys=("eval.matched_bench",),
        science=True,
    ),
    _r(
        "gravity_flop_ledger",
        OperatorClass.NUMERICAL_AUTHORITY,
        "FLOP-and-byte ledger separating compression from arithmetic savings.",
        handler_keys=("ledger.flop",),
        science=True,
    ),
    _r(
        "gravity_forge",
        OperatorClass.OPERATOR,
        "Capability-preserving sub-bit representation foundry.",
        handler_keys=("pack.gravity_forge",),
        science=True,
    ),
    _r(
        "gravity_format",
        OperatorClass.OPERATOR,
        "Native .gravity container format (header/shard read/write/verify).",
        handler_keys=("format.gravity",),
        science=True,
    ),
    _r(
        "gravity_functional_codec",
        OperatorClass.OPERATOR,
        "glm52.functional.moe.v1 codec storing a function, not weights.",
        handler_keys=("codec.functional_moe",),
        science=True,
    ),
    _r(
        "gravity_kernel_select",
        OperatorClass.NUMERICAL_AUTHORITY,
        "Kernel selection matrix: which grammar executes which geometry.",
        handler_keys=("select.kernel",),
        science=True,
    ),
    _r(
        "gravity_metal",
        OperatorClass.OPERATOR,
        "Hand-written Metal kernel: decode inside accumulation, never in memory.",
        handler_keys=("kernel.metal",),
        science=True,
    ),
    _r(
        "gravity_metal_lab_b",
        OperatorClass.OPERATOR,
        "Track B shared-table lookup-linear measured on unshared reality (VARIANT).",
        handler_keys=("kernel.metal_lab_b",),
        science=True,
    ),
    _r(
        "gravity_moe_layer",
        OperatorClass.OPERATOR,
        "Complete GLM-5.2 MoE layer as one Metal command buffer, parity-gated.",
        handler_keys=("forward.moe_layer",),
        science=True,
    ),
    _r(
        "gravity_real_fixtures",
        OperatorClass.FIXTURE,
        "Real packed tensors as fixtures from live campaign without disturbance.",
        handler_keys=("fixture.real_packed",),
        science=True,
    ),
    _r(
        "hawking_null_metric",
        OperatorClass.NUMERICAL_AUTHORITY,
        "Null-corrected promotion metric (constant-null vs raw cosine).",
        handler_keys=("metric.null",),
        science=True,
    ),
    # --- small shared helpers -------------------------------------------------
    _r(
        "bounded_cache",
        OperatorClass.OPERATOR,
        "Pressure-aware LRU for decoded experts / large reusable tensors.",
        handler_keys=("cache.pressure",),
        science=False,
    ),
    _r(
        "eco_common",
        OperatorClass.NUMERICAL_AUTHORITY,
        "Shared seal/hash/atomic helpers for Ecosystem Frontier scaffold.",
        handler_keys=("eco.common",),
        science=False,
    ),
)


class OperatorRegistry:
    """Lookup operators by module name or handler key; invoke by key."""

    def __init__(
        self,
        records: tuple[OperatorRecord, ...] | None = None,
        *,
        handlers: Mapping[str, Handler] | None = None,
    ) -> None:
        self.records: tuple[OperatorRecord, ...] = records or IRREDUCIBLE_MODULES
        self._by_module: dict[str, OperatorRecord] = {
            r.module: r for r in self.records
        }
        self._by_handler: dict[str, OperatorRecord] = {}
        for rec in self.records:
            for key in rec.handler_keys:
                self._by_handler[key] = rec
        self.handlers: dict[str, Handler] = dict(handlers or {})

    def get(self, module: str) -> OperatorRecord | None:
        return self._by_module.get(module)

    def for_handler(self, key: str) -> OperatorRecord | None:
        return self._by_handler.get(key)

    def register_handler(self, key: str, handler: Handler) -> None:
        self.handlers[key] = handler

    def resolve_handler(self, key: str) -> Handler | None:
        return self.handlers.get(key)

    def classification(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def unclassifiable(self) -> list[dict[str, Any]]:
        return [
            {
                "module": r.module,
                "loc": r.loc,
                "what_it_is": r.why,
            }
            for r in self.records
            if r.class_ is OperatorClass.UNCLASSIFIED
        ]

    def science_floor_loc(self) -> int:
        """LOC of modules that are live science (not pure path architecture)."""
        return sum(r.loc for r in self.records if r.science)

    def path_loc(self) -> int:
        """LOC of modules that are path machinery / shared non-science."""
        return sum(r.loc for r in self.records if not r.science)

    def summary(self) -> dict[str, Any]:
        by_class: dict[str, int] = {}
        loc_by_class: dict[str, int] = {}
        for r in self.records:
            by_class[r.class_name] = by_class.get(r.class_name, 0) + 1
            loc_by_class[r.class_name] = loc_by_class.get(r.class_name, 0) + r.loc
        return {
            "schema": "hawking.condense.operator_registry.v1",
            "module_count": len(self.records),
            "total_loc": sum(r.loc for r in self.records),
            "science_floor_loc": self.science_floor_loc(),
            "path_residue_loc": self.path_loc(),
            "by_class_count": by_class,
            "by_class_loc": loc_by_class,
            "unclassifiable": self.unclassifiable(),
            "path_sealed": [r.module for r in self.records if r.path_sealed],
        }


# Process-wide default registry (handlers filled by runtime builtins).
DEFAULT_REGISTRY = OperatorRegistry()


def load_default_registry(
    handlers: Mapping[str, Handler] | None = None,
) -> OperatorRegistry:
    return OperatorRegistry(handlers=handlers)


def classify_all() -> list[dict[str, Any]]:
    return DEFAULT_REGISTRY.classification()
