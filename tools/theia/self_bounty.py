"""Ingest local Hawking receipts as §19.12 HAWKING SELF-BOUNTY work.

Intake reads LOCAL artifacts only. Unknown schemas refuse rather than guess.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from tools.theia.bounty import (
    Bounty,
    BountyClass,
    Budget,
    PublicOrPrivate,
    hawking_internal_scope,
)
from tools.theia.intake import IntakeRefused, local_artifact
from tools.theia.labs import LabKind, SelfBountyKind
from tools.theia.value import DeclaredFactor, ValueInputs, ValueRefused


SCHEMA_KIND: dict[str, SelfBountyKind] = {
    "hawking.future.autonomy_scars.v1": SelfBountyKind.NEGATIVE_SCIENCE,
    "hawking.future.campaign_scars.v1": SelfBountyKind.NEGATIVE_SCIENCE,
    "hawking.future.complete_ebpw.v1": SelfBountyKind.REPRESENTATION_WIN,
    "hawking.future.device_compiler.v1": SelfBountyKind.NEW_COMPILER_PASS,
    "hawking.future.repro_science.v1": SelfBountyKind.REGRESSIONS,
}

SCHEMA_SECONDARY: dict[str, tuple[SelfBountyKind, ...]] = {
    "hawking.future.autonomy_scars.v1": (SelfBountyKind.AUTONOMY_RECOVERY_PROOF,),
}

VERIFIER = "tools.theia.intake.verify_receipt"

UNIT_SOURCE = (
    "declared unit baseline so the H.1 denominator is defined; STATIC_ONLY; "
    "not a hardware measurement and not complete_ebpw (wrong axes)"
)


def classify(doc: Mapping[str, Any]) -> SelfBountyKind:
    schema = doc.get("schema")
    if not isinstance(schema, str) or schema not in SCHEMA_KIND:
        raise IntakeRefused(
            f"no self-bounty mapping for schema {schema!r}; refusing rather than guessing"
        )
    return SCHEMA_KIND[schema]


def _unit(name: str, source: str) -> DeclaredFactor:
    return DeclaredFactor(value=Fraction(1), name=name, source=source)


def value_inputs_from_receipt(path: Path, doc: Mapping[str, Any]) -> ValueInputs:
    schema = doc.get("schema")
    if schema == "hawking.future.autonomy_scars.v1":
        scars = doc.get("scars")
        if not isinstance(scars, list) or not scars:
            raise ValueRefused("autonomy_scars receipt has no scars to ground information_gain")
        n = doc.get("n_scars")
        if n != len(scars):
            raise ValueRefused(f"n_scars {n} != len(scars) {len(scars)}")
        n_laws = sum(1 for s in scars if s.get("law"))
        if n_laws == 0 and doc.get("general_law"):
            n_laws = 1
        if n_laws == 0:
            raise ValueRefused("no law fields to ground transfer_value")
        return ValueInputs(
            verified_reward=_unit(
                "verified_reward",
                "H.1: verified_reward may include negative science; "
                f"schema {schema}",
            ),
            probability_of_success=_unit(
                "probability_of_success",
                f"artifact already on disk at {path}; intake is local parse, not a search",
            ),
            information_gain=DeclaredFactor(
                value=Fraction(int(n)),
                name="information_gain",
                source="receipt n_scars (count of recorded defects)",
            ),
            transfer_value=DeclaredFactor(
                value=Fraction(n_laws),
                name="transfer_value",
                source="count of scars that state a law",
            ),
            strategic_relevance=_unit(
                "strategic_relevance",
                "HAWKING SELF-BOUNTY laboratory, §19.12: negative science",
            ),
            wall_time=_unit("wall_time", UNIT_SOURCE),
            compute_cost=_unit("compute_cost", UNIT_SOURCE),
            human_cost=_unit("human_cost", UNIT_SOURCE),
            risk=_unit(
                "risk",
                "local readonly receipt; no network, no ACTIVE_TEST, no credentials",
            ),
            opportunity_cost=_unit("opportunity_cost", UNIT_SOURCE),
        )
    if schema == "hawking.future.complete_ebpw.v1":
        return ValueInputs(
            verified_reward=_unit(
                "verified_reward",
                "H.1: verified_reward may include a new compiler law / representation; "
                f"schema {schema}",
            ),
            probability_of_success=_unit(
                "probability_of_success",
                f"artifact already on disk at {path}",
            ),
            information_gain=_unit(
                "information_gain",
                "one complete-executable-BPW calculator law",
            ),
            transfer_value=_unit(
                "transfer_value",
                "refuse-missing-input doctrine reusable as a cost law",
            ),
            strategic_relevance=_unit(
                "strategic_relevance",
                "HAWKING SELF-BOUNTY laboratory, §19.12: representation win",
            ),
            wall_time=_unit("wall_time", UNIT_SOURCE),
            compute_cost=_unit("compute_cost", UNIT_SOURCE),
            human_cost=_unit("human_cost", UNIT_SOURCE),
            risk=_unit("risk", "local readonly receipt"),
            opportunity_cost=_unit("opportunity_cost", UNIT_SOURCE),
        )
    raise ValueRefused(f"no grounded H.1 mapping for schema {schema!r}")


def bounty_from_receipt(path: Path) -> tuple[Bounty, SelfBountyKind, dict[str, Any]]:
    artifact = local_artifact(str(path))
    doc = json.loads(artifact.read_text())
    if not isinstance(doc, dict):
        raise IntakeRefused(f"{artifact} is not a JSON object")
    kind = classify(doc)
    schema = str(doc.get("schema"))
    question = (
        doc.get("purpose")
        or doc.get("question")
        or doc.get("obligation")
        or artifact.name
    )
    bounty = Bounty(
        id=f"self:{artifact.stem}:{schema}",
        source=str(artifact.resolve()),
        domain="hawking",
        question_or_target=str(question),
        monetary_reward=None,
        nonmonetary_value=kind.value,
        authorization_scope=hawking_internal_scope(),
        rules=(
            "local artifact only",
            "no network",
            "no ACTIVE_TEST",
            "no training",
        ),
        evidence_required=("receipt schema", "seal_sha256"),
        verifier=VERIFIER,
        budget=Budget(workunits=1),
        deadline=None,
        public_or_private=PublicOrPrivate.PRIVATE,
        submission_policy="stage into receipts/future; do not publish externally",
        success_conditions=(
            "intake reaches TRAJECTORY + METHOD + NEGATIVE SCIENCE",
            "schedule score produced",
            "independent receipt verification",
        ),
        stop_conditions=("BLOCKED_RIGHTS", "missing receipt", "unreadable json"),
        bounty_class=BountyClass.HAWKING_INTERNAL_SELF_BOUNTY,
        lab=LabKind.HAWKING_SELF_BOUNTY.value,
    )
    return bounty, kind, doc
