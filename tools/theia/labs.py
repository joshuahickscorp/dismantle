"""§19.12 Theia laboratories as registered lab kinds.

AUTHORIZED SECURITY is registered so scope reasoning exists. It does not
gain an executable active-test path in this scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tools.theia.bounty import BountyClass, SECURITY_BOUNTY_CLASSES


class LabKind(Enum):
    MATH_FORMAL = "MATH / FORMAL"
    PHYSICS_QUANTUM = "PHYSICS / QUANTUM"
    SYSTEMS_COMPILER = "SYSTEMS / COMPILER"
    OPEN_SOURCE = "OPEN SOURCE"
    AUTHORIZED_SECURITY = "AUTHORIZED SECURITY"
    HAWKING_SELF_BOUNTY = "HAWKING SELF-BOUNTY"


assert len(LabKind) == 6, "§19.12 lists six laboratories"


class SelfBountyKind(Enum):
    """§19.12 HAWKING SELF-BOUNTY work items. Closed."""

    REGRESSIONS = "regressions"
    NEW_COMPILER_PASS = "new compiler pass"
    KERNEL_WIN = "kernel win"
    REPRESENTATION_WIN = "representation win"
    AUTONOMY_RECOVERY_PROOF = "autonomy/recovery proof"
    NEGATIVE_SCIENCE = "negative science"


assert len(SelfBountyKind) == 6


@dataclass(frozen=True)
class Lab:
    kind: LabKind
    work: tuple[str, ...]
    bounty_classes: tuple[BountyClass, ...]
    executable_work: tuple[str, ...]
    refused_work: tuple[str, ...]


LABS: dict[LabKind, Lab] = {
    LabKind.MATH_FORMAL: Lab(
        kind=LabKind.MATH_FORMAL,
        work=(
            "theorem_search",
            "formalization",
            "proof_repair",
            "counterexample_search",
            "symbolic_computation",
        ),
        bounty_classes=(
            BountyClass.FORMAL_PROOF_OR_COUNTEREXAMPLE,
            BountyClass.MATH_RESEARCH_CHALLENGE,
        ),
        executable_work=(),
        refused_work=(),
    ),
    LabKind.PHYSICS_QUANTUM: Lab(
        kind=LabKind.PHYSICS_QUANTUM,
        work=(
            "QM/QFT/GR/cosmology",
            "counterfactual_physical_law_research",
            "numerical_science",
            "observation_comparison",
        ),
        bounty_classes=(
            BountyClass.SCIENTIFIC_REPLICATION,
            BountyClass.SCIENTIFIC_ANOMALY_INVESTIGATION,
            BountyClass.PHYSICS_MODEL_DISCRIMINATION,
        ),
        executable_work=(),
        refused_work=(),
    ),
    LabKind.SYSTEMS_COMPILER: Lab(
        kind=LabKind.SYSTEMS_COMPILER,
        work=(
            "runtime_bugs",
            "compiler_bugs",
            "kernel_optimization",
            "performance_archaeology",
            "hardware_mapping",
        ),
        bounty_classes=(
            BountyClass.COMPILER_RUNTIME_BUG,
            BountyClass.GPU_KERNEL_OPTIMIZATION,
            BountyClass.PERFORMANCE_ENERGY_CHALLENGE,
        ),
        executable_work=(),
        refused_work=(),
    ),
    LabKind.OPEN_SOURCE: Lab(
        kind=LabKind.OPEN_SOURCE,
        work=(
            "issue_reproduction",
            "bug_localization",
            "tests",
            "patches",
            "upstream_quality_evidence",
        ),
        bounty_classes=(
            BountyClass.OPEN_SOURCE_BUG_ISSUE,
            BountyClass.PROTOCOL_TOOLING_INTEROPERABILITY,
            BountyClass.REPRODUCIBILITY_BOUNTY,
        ),
        executable_work=(),
        refused_work=(),
    ),
    LabKind.AUTHORIZED_SECURITY: Lab(
        kind=LabKind.AUTHORIZED_SECURITY,
        work=(
            "program_scope_reasoning",
            "safe_reproduction",
            "minimal_poc",
            "root_cause_analysis",
            "disclosure_package",
        ),
        bounty_classes=tuple(SECURITY_BOUNTY_CLASSES),
        executable_work=("program_scope_reasoning",),
        refused_work=(
            "ACTIVE_TEST",
            "safe_reproduction",
            "minimal_poc",
            "scan",
            "payload_generation",
            "network_egress",
            "credential_handling",
        ),
    ),
    LabKind.HAWKING_SELF_BOUNTY: Lab(
        kind=LabKind.HAWKING_SELF_BOUNTY,
        work=tuple(k.value for k in SelfBountyKind),
        bounty_classes=(BountyClass.HAWKING_INTERNAL_SELF_BOUNTY,),
        executable_work=tuple(k.value for k in SelfBountyKind),
        refused_work=("network_egress", "ACTIVE_TEST", "train_model"),
    ),
}


def lab(kind: LabKind) -> Lab:
    return LABS[kind]


def registered_lab_kinds() -> tuple[LabKind, ...]:
    return tuple(LabKind)
