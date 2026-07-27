"""Branch budgets, value scoring, and the stop rule that cannot be argued past.

Reuses `SearchEconomics` from `ramanujan.search` for the expansion counter. The layer
above it is the *branch* account: a named investigation line with a budget, a value
score, a halt reason, and a permanent record of why it stopped.

`odyssey/economics/ECONOMICS.json` sets the currency (verified evidence) and the reward
schedule. Refutation and clean exhaustion are paid deliberately: a system that only
pays for promotion learns to avoid falsification.

The economist never sees claim content -- that constraint is enforced in
`roles.RoleSession.write_budget`, not here. This module scores and stops; it does not
read statements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ramanujan.search import SearchEconomics, SearchResult, best_first, ProofState, Tactic


# From odyssey/economics/ECONOMICS.json. Kept as data, not magic numbers at call sites.
REWARDS: dict[str, float] = {
    "clean_branch_exhaustion": 3.0,
    "refutation": 5.0,
    "tier0_to_tier1": 1.0,
    "tier1_to_tier2": 3.0,
    "tier2_to_tier3": 10.0,
    "tribunal_admission": 25.0,
}

CURRENCY = "verified evidence"
ALLOCATOR = "economist"


class BranchHalted(RuntimeError):
    """Raised when a caller tries to spend on a branch that has already stopped."""


@dataclass
class HaltRecord:
    """Why a branch stopped. The reason is the whole point of recording it."""

    reason: str
    spent: int
    value_earned: float
    detail: dict = field(default_factory=dict)


@dataclass
class BranchAccount:
    """A named branch with a budget the search cannot talk its way past.

    When the budget is exhausted the branch HALTS and records why. A halted branch
    refuses further charges. That is the stop rule: not a suggestion, not a metric
    that a later stage can ignore.
    """

    branch_id: str
    economics: SearchEconomics = field(default_factory=SearchEconomics)
    value_earned: float = 0.0
    halt: HaltRecord | None = None
    charges: list[dict] = field(default_factory=list)
    value_events: list[dict] = field(default_factory=list)

    def is_halted(self) -> bool:
        return self.halt is not None

    def may_spend(self) -> bool:
        if self.is_halted():
            return False
        return self.economics.may_expand()

    def why_cannot_spend(self) -> str:
        if self.halt is not None:
            return f"already_halted: {self.halt.reason}"
        if not self.economics.may_expand():
            return "economics: max_expansions"
        return "ok"

    def charge(self, units: int = 1, what: str = "expand") -> None:
        """Spend budget. Halts and records when the budget is exhausted mid-charge."""
        if self.is_halted():
            raise BranchHalted(
                f"branch {self.branch_id!r} is halted ({self.halt.reason}); no further charges"
            )
        for _ in range(units):
            if not self.economics.may_expand():
                self.stop("economics: max_expansions", detail={"last_attempt": what})
                raise BranchHalted(
                    f"branch {self.branch_id!r} exhausted budget after {self.economics.spent} expansions"
                )
            self.economics.charge()
            self.charges.append({"what": what, "spent_after": self.economics.spent})

    def stop(self, reason: str, detail: dict | None = None) -> HaltRecord:
        """Halt the branch and record why. Idempotent on reason if already halted."""
        if self.halt is not None:
            return self.halt
        self.halt = HaltRecord(
            reason=reason,
            spent=self.economics.spent,
            value_earned=self.value_earned,
            detail=dict(detail or {}),
        )
        return self.halt

    def award(self, kind: str, detail: dict | None = None) -> float:
        """Credit value for a verified outcome. Unknown kinds earn nothing (fail closed)."""
        if kind not in REWARDS:
            raise ValueError(
                f"unknown value kind {kind!r}; currency is {CURRENCY!r} and the "
                f"schedule is fixed: {sorted(REWARDS)}"
            )
        amount = REWARDS[kind]
        self.value_earned += amount
        self.value_events.append({"kind": kind, "amount": amount, "detail": detail or {}})
        return amount

    def score(self) -> dict:
        """Value-per-spend summary. Clean exhaustion is itself a reward-eligible outcome."""
        spent = max(1, self.economics.spent)
        return {
            "branch_id": self.branch_id,
            "spent": self.economics.spent,
            "value_earned": self.value_earned,
            "value_per_spend": self.value_earned / spent,
            "halted": self.is_halted(),
            "halt_reason": None if self.halt is None else self.halt.reason,
            "currency": CURRENCY,
        }


def run_branch_search(
    branch: BranchAccount,
    start: ProofState,
    tactics: Tactic,
    heuristic: Callable[[ProofState], float],
) -> tuple[SearchResult, BranchAccount]:
    """Best-first search bound to a branch budget.

    On budget exhaustion the SearchResult already carries `stopped_by`; this also
    writes a HaltRecord onto the branch so the stop is visible outside the search
    stack (Ledger-adjacent callers, the economist, the Cartographer).
    """
    result, _dag = best_first(start, tactics, heuristic, economics=branch.economics)
    if result.stopped_by.startswith("economics:"):
        branch.stop(result.stopped_by, detail={"found": result.found, "path_len": len(result.path)})
        # Clean exhaustion is a positive outcome under the reward schedule when the
        # branch did not find a proof -- it stopped for a real reason rather than
        # wandering. Award only on genuine budget halt, not on depth cutoffs.
        if not result.found and result.stopped_by == "economics: max_expansions":
            branch.award("clean_branch_exhaustion", detail={"expansions": result.expansions})
    elif result.stopped_by == "exhausted":
        branch.stop("search: exhausted", detail={"expansions": result.expansions})
        if not result.found:
            branch.award("clean_branch_exhaustion", detail={"expansions": result.expansions})
    elif result.stopped_by == "closed":
        # Finding a fixture goal is not Tier-3; do not award promotion value here.
        branch.stop("search: closed", detail={"path": list(result.path)})
    return result, branch


def value_for_tier_step(from_tier: int, to_tier: int) -> str | None:
    """Map a tier transition to a reward kind, or None if the step is not paid."""
    key = f"tier{from_tier}_to_tier{to_tier}"
    return key if key in REWARDS else None
