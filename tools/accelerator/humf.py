"""HUMF — the Hawking Unified Memory Fabric. FRONT H (G050, steer S015).

HUMF DOES NOT LIE. Apple unified memory plus external VRAM is never called
physically unified memory. The physical domains stay distinct and HUMF manages the
illusion at the object level: who owns the current state, which copies are valid,
what a move would cost, and whether recomputing beats transferring.

Every number a MOCK domain produces is stamped SIMULATED. The steer is explicit
that a simulated transport number must never be labelled physical evidence, so the
planner carries the provenance of each cost it used into its decision, and a plan
built on simulated numbers says so in its own output rather than in a footnote.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class State(str, Enum):
    """Steer §14. No accidental hidden coherence assumptions."""
    ABSENT = "ABSENT"
    CLEAN = "CLEAN"
    DIRTY = "DIRTY"
    STALE = "STALE"
    MATERIALIZING = "MATERIALIZING"
    TRANSFERRING = "TRANSFERRING"
    INVALID = "INVALID"
    EVICTED = "EVICTED"


# Illegal transitions fail closed (the steer says so of the driver state machine;
# the same discipline belongs here, where a silent bad transition means silent
# data corruption rather than a crash).
LEGAL: dict[State, set[State]] = {
    State.ABSENT:        {State.MATERIALIZING, State.TRANSFERRING},
    State.MATERIALIZING: {State.CLEAN, State.INVALID},
    State.TRANSFERRING:  {State.CLEAN, State.INVALID},
    State.CLEAN:         {State.DIRTY, State.STALE, State.EVICTED, State.INVALID,
                          State.TRANSFERRING},
    State.DIRTY:         {State.CLEAN, State.STALE, State.INVALID},
    State.STALE:         {State.TRANSFERRING, State.MATERIALIZING, State.EVICTED,
                          State.INVALID},
    State.EVICTED:       {State.MATERIALIZING, State.TRANSFERRING, State.ABSENT},
    State.INVALID:       {State.ABSENT, State.MATERIALIZING},
}


class HumfError(RuntimeError):
    pass


@dataclass
class Domain:
    """A memory domain. `physical` is False for anything mocked, and that flag is
    what stops a simulated bandwidth from being quoted as evidence."""
    name: str
    bytes_capacity: int
    bandwidth_gb_s: float
    physical: bool
    latency_s: float = 0.0

    @property
    def provenance(self) -> str:
        return "MEASURED" if self.physical else "SIMULATED"


@dataclass
class Materialization:
    """One representation of one object in one domain."""
    domain: str
    representation: str
    layout: str
    bytes: int
    state: State = State.ABSENT
    payload: bytes | None = None      # the actual data, when this copy holds any

    def transition(self, to: State) -> None:
        if to not in LEGAL[self.state]:
            raise HumfError(f"illegal transition {self.state.value} -> {to.value} "
                            f"for {self.representation} in {self.domain}")
        self.state = to


@dataclass
class HumfObject:
    """The twelve recorded fields of steer §32, as one object across domains."""
    identity: str                                   # 1 ObjectIdentity
    logical_type: str                               # 2 logical type
    elements: int
    dtype: str
    materializations: dict[str, Materialization] = field(default_factory=dict)  # 3,7,8,9
    owner: str | None = None                        # 4 current owner of live state
    recompute_cost_s: float | None = None           # 11
    next_use_hint: str | None = None                # 12
    recompute: Callable[[], Any] | None = None

    # 5 valid copies
    def valid_copies(self) -> list[str]:
        return [d for d, m in self.materializations.items() if m.state is State.CLEAN]

    # 6 dirty state
    def is_dirty(self) -> bool:
        return any(m.state is State.DIRTY for m in self.materializations.values())

    def place(self, m: Materialization) -> None:
        self.materializations[m.domain] = m

    def mark_written(self, domain: str) -> None:
        """A write makes this copy DIRTY and every other copy STALE. Nothing about
        that is implicit."""
        if domain not in self.materializations:
            raise HumfError(f"{self.identity} has no materialization in {domain}")
        self.materializations[domain].transition(State.DIRTY)
        self.owner = domain
        for d, m in self.materializations.items():
            if d != domain and m.state is State.CLEAN:
                m.transition(State.STALE)


@dataclass
class Plan:
    action: str
    cost_s: float
    detail: str
    cost_provenance: str
    options: list[dict[str, Any]]

    @property
    def rests_on_simulated_numbers(self) -> bool:
        return self.cost_provenance != "MEASURED"


class Humf:
    def __init__(self, domains: dict[str, Domain],
                 providers: dict[str, Any] | None = None):
        self.domains = domains
        self.objects: dict[str, HumfObject] = {}
        self.log: list[dict[str, Any]] = []
        # Domains backed by a provider actually move bytes on execute(). Without
        # this the executor transitioned TRANSFERRING -> CLEAN on bookkeeping alone
        # and marked a copy valid that held nothing -- see the defect recorded in
        # ACCELERATOR_HUMF_FAILURE_INJECTION.json.
        self.providers: dict[str, Any] = providers or {}

    def register(self, obj: HumfObject) -> None:
        self.objects[obj.identity] = obj

    def _transfer_cost(self, src: str, dst: str, nbytes: int) -> tuple[float, str]:
        a, b = self.domains[src], self.domains[dst]
        bw = min(a.bandwidth_gb_s, b.bandwidth_gb_s) * 1e9
        prov = "MEASURED" if (a.physical and b.physical) else "SIMULATED"
        return nbytes / bw + a.latency_s + b.latency_s, prov

    def plan_acquire(self, identity: str, want_domain: str,
                     want_representation: str | None = None) -> Plan:
        """Steer §34: MOVE, COPY, RECOMPUTE, MATERIALIZE A DIFFERENT REPRESENTATION,
        or LEAVE IT AND MOVE THE COMPUTE. All estimated, cheapest wins."""
        obj = self.objects[identity]
        opts: list[dict[str, Any]] = []

        here = obj.materializations.get(want_domain)
        if here and here.state is State.CLEAN and (
                want_representation in (None, here.representation)):
            return Plan("ALREADY_RESIDENT", 0.0, f"{identity} is CLEAN in {want_domain}",
                        "MEASURED", [])

        for src, m in obj.materializations.items():
            if src == want_domain or m.state is not State.CLEAN:
                continue
            c, prov = self._transfer_cost(src, want_domain, m.bytes)
            opts.append({"action": "TRANSFER", "from": src, "bytes": m.bytes,
                         "representation": m.representation, "cost_s": c,
                         "cost_provenance": prov})

        if obj.recompute_cost_s is not None:
            opts.append({"action": "RECOMPUTE", "from": None,
                         "cost_s": obj.recompute_cost_s,
                         "cost_provenance": "MEASURED"})

        if not opts:
            return Plan("IMPOSSIBLE", float("inf"),
                        f"{identity} has no CLEAN copy and no recompute recipe",
                        "MEASURED", [])

        best = min(opts, key=lambda o: o["cost_s"])
        return Plan(best["action"], best["cost_s"],
                    f"{identity} -> {want_domain} via {best['action']}"
                    + (f" from {best['from']}" if best.get("from") else ""),
                    best["cost_provenance"], opts)

    def execute(self, identity: str, plan: Plan, want_domain: str,
                representation: str = "dense_f32", layout: str = "row_major",
                nbytes: int | None = None) -> Materialization:
        obj = self.objects[identity]
        if plan.action == "ALREADY_RESIDENT":
            return obj.materializations[want_domain]
        if plan.action == "IMPOSSIBLE":
            raise HumfError(plan.detail)
        dst = obj.materializations.get(want_domain)
        if dst is None:
            src_any = next(iter(obj.materializations.values()), None)
            dst = Materialization(want_domain, representation, layout,
                                  nbytes if nbytes is not None
                                  else (src_any.bytes if src_any else 0))
            obj.place(dst)
        dst.transition(State.TRANSFERRING if plan.action == "TRANSFER"
                       else State.MATERIALIZING)
        # MOVE THE BYTES. A transfer that is only a state change cannot fail, and a
        # transfer that cannot fail is not a transfer -- it is an assumption. If the
        # provider raises, the destination goes INVALID and stays out of
        # valid_copies(); the SOURCE is left untouched, because a failed outbound
        # transfer must not damage the copy that was good.
        try:
            self._move(obj, plan, want_domain, dst)
        except Exception:
            dst.transition(State.INVALID)
            self.log.append({"object": identity, "action": plan.action,
                             "domain": want_domain, "outcome": "FAILED",
                             "at": time.time()})
            raise
        dst.transition(State.CLEAN)
        if obj.owner is None:
            obj.owner = want_domain
        self.log.append({"object": identity, "action": plan.action,
                         "domain": want_domain, "cost_s": plan.cost_s,
                         "cost_provenance": plan.cost_provenance,
                         "outcome": "OK", "at": time.time()})
        return dst

    def _move(self, obj: "HumfObject", plan: Plan, want_domain: str,
              dst: Materialization) -> None:
        """Actually put the data where the plan says it goes.

        RECOMPUTE runs the object's recipe. TRANSFER reads a valid source copy and
        hands it to the destination domain's provider, if that domain has one. A
        domain with no provider still only gets bytes it was given -- it never
        gets a payload conjured from nothing.
        """
        if plan.action == "RECOMPUTE":
            if obj.recompute is None:
                raise HumfError(f"{obj.identity} has no recompute recipe")
            dst.payload = obj.recompute()
            return
        src = next((m for d, m in obj.materializations.items()
                    if d != want_domain and m.state is State.CLEAN), None)
        if src is None or src.payload is None:
            raise HumfError(
                f"transfer of {obj.identity} into {want_domain} has no valid source "
                f"payload. A STALE or empty copy is never a transfer source.")
        prov = self.providers.get(want_domain)
        if prov is not None:
            prov.copy_in(obj.identity, src.payload)
            dst.payload = prov.copy_out(obj.identity)
        else:
            dst.payload = src.payload


class MockExternalMemoryProvider:
    """Proves the HUMF state machine, NOT GPU performance. Its bandwidth is a knob,
    not a measurement, and every domain it hands out is physical=False."""

    def __init__(self, *, capacity_bytes: int, bandwidth_gb_s: float,
                 latency_s: float = 0.0):
        self.domain = Domain("MOCK_EXTERNAL_VRAM", capacity_bytes, bandwidth_gb_s,
                             physical=False, latency_s=latency_s)
        self.allocated = 0
        self.fail_next: str | None = None      # failure injection
        # Real bytes, so a round trip proves the DATA survived rather than only the
        # bookkeeping agreeing with itself.
        self.store: dict[str, bytes] = {}

    def allocate(self, nbytes: int) -> int:
        if self.fail_next == "allocate":
            self.fail_next = None
            raise HumfError("injected failure: allocate")
        if self.allocated + nbytes > self.domain.bytes_capacity:
            raise HumfError(f"out of capacity: {self.allocated + nbytes} > "
                            f"{self.domain.bytes_capacity}")
        self.allocated += nbytes
        return nbytes

    def free(self, nbytes: int) -> None:
        self.allocated = max(0, self.allocated - nbytes)

    def copy_in(self, key: str, payload: bytes) -> None:
        if self.fail_next == "copy":
            self.fail_next = None
            raise HumfError("injected failure: copy_in")
        self.store[key] = bytes(payload)

    def copy_out(self, key: str) -> bytes:
        if self.fail_next == "copy":
            self.fail_next = None
            raise HumfError("injected failure: copy_out")
        if key not in self.store:
            raise HumfError(f"{key} is not resident in {self.domain.name}")
        return self.store[key]
