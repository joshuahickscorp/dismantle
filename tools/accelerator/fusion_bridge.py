"""Fusion Bridge -- compose SemanticTransportEdge + Placement into a
heterogeneous execution plan.

The three device-neutral pieces:

    semantic_transport.SemanticTransportEdge   typed inter-domain edge
    placement.Placement                        node-to-domain assignment
    this module                                plan + validator + cost

Vendor specifics belong in backends. This file has no product branch.

WHAT THIS CONNECTS TO, RATHER THAN REIMPLEMENTING:

  fusion_planner.Topology / Link / Route
      Cost routing. A Route's MEASURED/SIMULATED hop flag is mapped to
      Cost.all_hops_physical; the Cost.label is still COST_MODEL. The
      planner's object-home Placement is lifted into placement.Placement
      (STORAGE nodes) via lift_object_placements().

  fusion_isa.FusionTimeline
      Edges and placements become ACQUIRE / COPY / FENCE / SUBMIT /
      RELEASE commands. Bookkeeping only; no backend.

  humf.MemoryClass / MEMORY_CLASS_POLICY
      Mutability for lifted object-home placements. Trust/state stay in
      HUMF; this module does not reimplement the coherence engine.

  hcli.physical_graph.PhysicalGraph  (READ, never modified)
      overlay_physical_graph() copies a compiled graph dict and fills
      device_placement / synchronization / dependencies from the plan.

  tools.future.hwir
      FPGA_HBM placements lower to HwirNode; inter-domain edges into that
      domain become dma-transport nodes whose primitive is the atlas name
      SemanticTransportEdge (already in hwir.PRIMITIVE_TO_NODE_KIND).

  machine_genome.build() payload
      domain_from_machine_genome() reads capacity facts already gathered.
      It does not call measure_bandwidth().

  tools/odyssey/device_profiles.py
      INTERACTIVE / MAXX are workload profiles, not device domains.
      Placement.profile_hint admits those two names and nothing else.

COST LAW. Every cost this module emits is labeled COST_MODEL. No path
assigns HARDWARE_MEASURED. A declared FPGA_HBM domain is a cost-model
domain, never a measurement.

VALIDATOR LAW (roadmap §16.1 / §16.2):
  - assumption rank > link rank  ->  COHERENCY_OVERCLAIM
  - HARDWARE_UMA on a multi-domain edge  ->  HARDWARE_UMA_ACROSS_DOMAINS
  - device_compute evidenced only by readback  ->  READBACK_IS_NOT_COMPUTE_VISIBILITY
  - any Cost.label other than COST_MODEL  ->  COST_LABEL_REFUSED
The coherency check is _reject_coherency_overclaim; that is the mutation
point the negative test is built on.

NOT IMPLEMENTED, named rather than left silent:
  - No real transport, no FPGA bitstream, no discrete-GPU runtime.
  - No persistent token pipeline (roadmap §15.9). token_id is metadata.
  - No online replanning. A HeterogeneousPlan is a point-in-time estimate.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from placement import (
    NodeKind,
    Placement,
    ResourceRequirements,
    ValidityCondition,
    kind_admits,
    resources_fit,
)
from semantic_transport import (
    ATLAS_PRIMITIVE,
    COST_MODEL,
    HARDWARE_MEASURED,
    HWIR_NODE_KIND,
    CoherencyAssumption,
    ComputeVisibilityEvidence,
    Cost,
    DomainKind,
    DomainVisibility,
    ExecutionDomain,
    OwnershipTransfer,
    PayloadSemantics,
    SemanticTransportEdge,
    SyncRequirement,
    cost_model,
)

import fusion_planner as fp
from fusion_isa import FusionCommand, FusionOp, FusionTimeline
from humf import MemoryClass


SCHEMA = "hawking.accelerator.fusion_bridge.v1"
QUALIFICATION = "PLAN_ONLY"

# Declared-domain interconnect knobs. Same order of magnitude as the
# external-bridge figures fusion_planner already uses for structural tests.
# Not a measurement. Not a datasheet claim.
DECLARED_BRIDGE_BW_GB_S = 12.0
DECLARED_BRIDGE_LATENCY_S = 2.5e-4

# Demo domain names. Kinds, not products: GPU_UMA is the host unified-memory
# GPU domain; FPGA_HBM is a declared future domain. Tests bind the first to
# the host GPU and the second to a not-present FPGA.
HOST_GPU_UMA = "gpu_uma_0"
DECLARED_FPGA_HBM = "fpga_hbm_0"


class FusionBridgeError(RuntimeError):
    """Base for every error this module raises."""


@dataclass
class ValidationReport:
    ok: bool
    errors: list[dict[str, str]]

    def codes(self) -> list[str]:
        return [e["code"] for e in self.errors]

    def to_dict(self) -> dict:
        return {"errors": list(self.errors), "ok": self.ok}


def _err(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message, "path": path}


@dataclass
class HeterogeneousPlan:
    """A plan across two or more execution domains. Construction does not
    validate -- a deliberately illegal plan must be constructible so the
    negative test can exist. compose() validates and refuses."""
    domains: dict[str, ExecutionDomain]
    placements: tuple[Placement, ...]
    edges: tuple[SemanticTransportEdge, ...]
    object_placements: tuple[fp.Placement, ...] = ()
    notes: tuple[str, ...] = ()
    qualification: str = QUALIFICATION

    def to_dict(self) -> dict:
        return {
            "domains": {k: v.to_dict() for k, v in sorted(self.domains.items())},
            "edges": [e.to_dict() for e in self.edges],
            "notes": list(self.notes),
            "object_placements": [
                {"home": p.home, "identity": p.identity,
                 "reason": p.reason, "replicas": list(p.replicas)}
                for p in self.object_placements
            ],
            "placements": [p.to_dict() for p in self.placements],
            "qualification": self.qualification,
            "schema": SCHEMA,
        }

    def total_cost(self) -> Cost:
        """Sum of edge costs. Always COST_MODEL. Intra-domain work is not
        estimated here (no kernel model); this is transport cost only."""
        time_s = sum(e.cost.time_s for e in self.edges)
        nbytes = sum(e.cost.nbytes for e in self.edges)
        hops_physical = bool(self.edges) and all(e.cost.all_hops_physical for e in self.edges)
        return cost_model(
            time_s=time_s,
            nbytes=nbytes,
            bandwidth_gb_s=0.0 if not self.edges else min(e.cost.bandwidth_gb_s for e in self.edges),
            latency_s=sum(e.cost.latency_s for e in self.edges),
            note="sum of edge COST_MODEL estimates; not a hardware measurement",
            all_hops_physical=hops_physical,
        )


def cost_from_route(route: fp.Route, *, note: str = "") -> Cost:
    """Wrap a fusion_planner.Route. The planner stamps MEASURED when every
    hop is physical=True; that flag is preserved as all_hops_physical and
    does NOT become HARDWARE_MEASURED. This layer's label is COST_MODEL."""
    bw = 0.0
    lat = route.alpha_s
    if route.hops:
        bw = min(h.bandwidth_gb_s for h in route.hops)
    return cost_model(
        time_s=route.total_time_s,
        nbytes=route.nbytes,
        bandwidth_gb_s=bw,
        latency_s=lat,
        note=note or "from fusion_planner.Route; COST_MODEL",
        all_hops_physical=all(h.physical for h in route.hops) if route.hops else True,
    )


def declared_heterogeneous_topology() -> fp.Topology:
    """Host GPU_UMA + a declared FPGA_HBM domain, linked by a non-physical
    interconnect. The FPGA domain is not present on this machine."""
    t = fp.Topology()
    t.add_domain(HOST_GPU_UMA, physical=True)
    t.add_domain(DECLARED_FPGA_HBM, physical=False)
    t.add_link(
        HOST_GPU_UMA, DECLARED_FPGA_HBM,
        bandwidth_gb_s=DECLARED_BRIDGE_BW_GB_S,
        latency_s=DECLARED_BRIDGE_LATENCY_S,
        physical=False,
        note="declared host-to-fpga interconnect; COST_MODEL knob",
    )
    return t


def host_gpu_uma_domain(*, capacity_bytes: int | None = None) -> ExecutionDomain:
    return ExecutionDomain(
        name=HOST_GPU_UMA,
        kind=DomainKind.GPU_UMA,
        physical=True,
        capacity_bytes=capacity_bytes,
        visibility=DomainVisibility(
            readback=True,
            device_compute=True,
            device_compute_evidence=ComputeVisibilityEvidence.EXPLICIT_CONTRACT,
        ),
        internal_coherency=CoherencyAssumption.HARDWARE_UMA,
    )


def declared_fpga_hbm_domain(*, capacity_bytes: int | None = None) -> ExecutionDomain:
    """A future FPGA_HBM domain. physical=False. Internal coherency is
    SOFTWARE_MANAGED (on-chip fabric); the link TO it is independently
    NONE -- that distinction is the whole point of SemanticTransportEdge."""
    return ExecutionDomain(
        name=DECLARED_FPGA_HBM,
        kind=DomainKind.FPGA_HBM,
        physical=False,
        capacity_bytes=capacity_bytes,
        visibility=DomainVisibility(
            readback=True,
            device_compute=True,
            device_compute_evidence=ComputeVisibilityEvidence.EXPLICIT_CONTRACT,
        ),
        internal_coherency=CoherencyAssumption.SOFTWARE_MANAGED,
    )


def lift_object_placements(
    topo: fp.Topology,
    objects: Sequence[fp.SemanticObject],
) -> tuple[tuple[Placement, ...], tuple[fp.Placement, ...]]:
    """Connect: fusion_planner.place_objects (object-home) -> Placement
    (STORAGE nodes). Replicas are copied through; mutability still comes
    from humf.MEMORY_CLASS_POLICY, which place_objects already consults."""
    raw = fp.place_objects(topo, objects)
    nbytes = {o.identity: o.nbytes for o in objects}
    lifted: list[Placement] = []
    for identity, p in raw.items():
        lifted.append(Placement(
            node_id=identity,
            node_kind=NodeKind.STORAGE,
            domain=p.home,
            resources=ResourceRequirements(bytes=nbytes.get(identity, 0)),
            validity=ValidityCondition(reason=p.reason),
            replicas=p.replicas,
        ))
    return tuple(lifted), tuple(raw.values())


def domain_from_machine_genome(
    genome: Mapping[str, Any],
    *,
    name: str = HOST_GPU_UMA,
    kind: DomainKind = DomainKind.GPU_UMA,
) -> ExecutionDomain:
    """Connect: MachineGenome facts -> ExecutionDomain. Reads capacity
    already recorded; does not measure. A genome without memory_bytes
    yields capacity_bytes=None (undeclared, not zero)."""
    mem = genome.get("memory_bytes")
    capacity = int(mem) if mem else None
    internal = (CoherencyAssumption.HARDWARE_UMA if kind is DomainKind.GPU_UMA
                else CoherencyAssumption.SOFTWARE_MANAGED)
    return ExecutionDomain(
        name=name,
        kind=kind,
        physical=True,
        capacity_bytes=capacity,
        visibility=DomainVisibility(
            readback=True,
            device_compute=True,
            device_compute_evidence=ComputeVisibilityEvidence.EXPLICIT_CONTRACT,
        ),
        internal_coherency=internal,
    )


def edge_cost_on(
    topo: fp.Topology,
    source: str,
    destination: str,
    nbytes: int,
    *,
    note: str = "",
) -> Cost:
    route = topo.shortest_path(source, destination, nbytes)
    return cost_from_route(route, note=note)


# --------------------------------------------------------------------------- validator
#
# _reject_coherency_overclaim is the mutation point. The negative test
# constructs a plan whose coherency_assumption outranks link_coherency and
# asserts that validate() returns COHERENCY_OVERCLAIM. Removing the call
# below must make that test FAIL; restoring it must make the test PASS.


def _reject_coherency_overclaim(plan: HeterogeneousPlan, errors: list[dict[str, str]]) -> None:
    """Refuse a plan that assumes stronger coherency than the link declares.

    Roadmap §16.2: never falsely claim rung 6 (HARDWARE_UMA) across devices.
    Roadmap §16.1: the check is about what the plan CLAIMS, not about a
    readback that happened to match.
    """
    for i, edge in enumerate(plan.edges):
        path = f"edges[{i}]"
        if edge.assumes_stronger_than_link:
            errors.append(_err(
                "COHERENCY_OVERCLAIM",
                path,
                f"edge {edge.source}->{edge.destination} assumes "
                f"{edge.coherency_assumption.name} (rank {int(edge.coherency_assumption)}) "
                f"on a link declared {edge.link_coherency.name} "
                f"(rank {int(edge.link_coherency)})",
            ))
        if (edge.source != edge.destination
                and edge.coherency_assumption is CoherencyAssumption.HARDWARE_UMA):
            errors.append(_err(
                "HARDWARE_UMA_ACROSS_DOMAINS",
                path,
                f"edge {edge.source}->{edge.destination} assumes HARDWARE_UMA "
                f"across distinct domains; rung 6 is intra-GPU_UMA only",
            ))


def _reject_readback_as_compute_visibility(
    plan: HeterogeneousPlan, errors: list[dict[str, str]],
) -> None:
    """§16.1: READBACK CORRECTNESS DOES NOT PROVE DEVICE-COMPUTE VISIBILITY."""
    for name, dom in plan.domains.items():
        vis = dom.visibility
        if vis.device_compute and vis.device_compute_evidence is ComputeVisibilityEvidence.READBACK:
            errors.append(_err(
                "READBACK_IS_NOT_COMPUTE_VISIBILITY",
                f"domains.{name}.visibility",
                f"domain {name!r} claims device_compute on READBACK evidence; "
                f"readback correctness does not prove device-compute visibility",
            ))
        if vis.device_compute and vis.device_compute_evidence is ComputeVisibilityEvidence.UNDECLARED:
            errors.append(_err(
                "COMPUTE_VISIBILITY_UNDECLARED",
                f"domains.{name}.visibility",
                f"domain {name!r} claims device_compute with UNDECLARED evidence; "
                f"fail closed",
            ))


def _reject_bad_costs(plan: HeterogeneousPlan, errors: list[dict[str, str]]) -> None:
    costs: list[tuple[str, Cost]] = [(f"edges[{i}].cost", e.cost) for i, e in enumerate(plan.edges)]
    costs.append(("total_cost", plan.total_cost()))
    for path, cost in costs:
        if cost.label != COST_MODEL:
            errors.append(_err(
                "COST_LABEL_REFUSED",
                path,
                f"cost label {cost.label!r} is not {COST_MODEL!r}",
            ))
        if cost.label == HARDWARE_MEASURED:
            errors.append(_err(
                "HARDWARE_MEASURED_EMITTED",
                path,
                "HARDWARE_MEASURED is forbidden in this layer",
            ))


def _reject_placement_errors(plan: HeterogeneousPlan, errors: list[dict[str, str]]) -> None:
    seen: set[str] = set()
    for i, p in enumerate(plan.placements):
        path = f"placements[{i}]"
        if p.node_id in seen:
            errors.append(_err("DUPLICATE_NODE", path, f"node_id {p.node_id!r} placed twice"))
        seen.add(p.node_id)
        if p.domain not in plan.domains:
            errors.append(_err(
                "UNKNOWN_DOMAIN",
                f"{path}.domain",
                f"placement {p.node_id!r} names domain {p.domain!r} which is not in the plan",
            ))
            continue
        dom = plan.domains[p.domain]
        if not kind_admits(dom.kind, p.node_kind):
            errors.append(_err(
                "KIND_NOT_ADMITTED",
                path,
                f"{p.node_kind.value} is not admitted on {dom.kind.value}",
            ))
        if not resources_fit(p.resources, dom.capacity_bytes):
            errors.append(_err(
                "RESOURCES_EXCEED_CAPACITY",
                path,
                f"{p.node_id!r} asks {p.resources.bytes}B; domain {dom.name!r} "
                f"capacity is {dom.capacity_bytes}B",
            ))
        if p.node_kind is NodeKind.STATE and not p.owner:
            errors.append(_err(
                "STATE_NO_OWNER",
                path,
                f"state node {p.node_id!r} has no owner domain",
            ))
        for r in p.replicas:
            if r not in plan.domains:
                errors.append(_err(
                    "UNKNOWN_DOMAIN",
                    f"{path}.replicas",
                    f"replica domain {r!r} is not in the plan",
                ))


def _reject_edge_errors(plan: HeterogeneousPlan, errors: list[dict[str, str]]) -> None:
    for i, edge in enumerate(plan.edges):
        path = f"edges[{i}]"
        for end, label in ((edge.source, "source"), (edge.destination, "destination")):
            if end not in plan.domains:
                errors.append(_err(
                    "UNKNOWN_DOMAIN",
                    f"{path}.{label}",
                    f"edge {label} {end!r} is not in the plan",
                ))
        if edge.cost.label != COST_MODEL:
            errors.append(_err(
                "COST_LABEL_REFUSED",
                f"{path}.cost",
                f"edge cost label {edge.cost.label!r} is not {COST_MODEL!r}",
            ))


def validate(plan: HeterogeneousPlan) -> ValidationReport:
    errors: list[dict[str, str]] = []
    if len(plan.domains) < 2:
        errors.append(_err(
            "NOT_HETEROGENEOUS",
            "domains",
            "a fusion-bridge plan needs at least two execution domains",
        ))
    _reject_placement_errors(plan, errors)
    _reject_edge_errors(plan, errors)
    _reject_readback_as_compute_visibility(plan, errors)
    _reject_bad_costs(plan, errors)
    _reject_coherency_overclaim(plan, errors)  # MUTATION_POINT: coherency check
    errors.sort(key=lambda e: (e["code"], e["path"], e["message"]))
    return ValidationReport(ok=not errors, errors=errors)


def compose(
    domains: Mapping[str, ExecutionDomain],
    placements: Sequence[Placement],
    edges: Sequence[SemanticTransportEdge],
    *,
    object_placements: Sequence[fp.Placement] = (),
    notes: Sequence[str] = (),
) -> HeterogeneousPlan:
    """Build and refuse unless validate() is clean. Illegal plans for the
    negative test must be constructed via HeterogeneousPlan(...) directly."""
    plan = HeterogeneousPlan(
        domains=dict(domains),
        placements=tuple(placements),
        edges=tuple(edges),
        object_placements=tuple(object_placements),
        notes=tuple(notes),
    )
    report = validate(plan)
    if not report.ok:
        codes = ", ".join(report.codes())
        raise FusionBridgeError(f"plan refused: {codes}")
    return plan


# --------------------------------------------------------------------------- demo plan (valid two-domain)


def two_domain_plan(
    *,
    activation_bytes: int = 1 << 20,
    weight_bytes: int = 8 << 20,
    host_capacity: int | None = None,
    fpga_capacity: int | None = None,
) -> HeterogeneousPlan:
    """Host GPU_UMA + declared FPGA_HBM, non-coherent link, COST_MODEL
    transport of activations, weights resident on the FPGA domain, KV
    state resident on the host. This is the acceptance happy path."""
    topo = declared_heterogeneous_topology()
    host = host_gpu_uma_domain(capacity_bytes=host_capacity)
    fpga = declared_fpga_hbm_domain(capacity_bytes=fpga_capacity)
    domains = {host.name: host, fpga.name: fpga}

    objects = [
        fp.SemanticObject(
            "weights", MemoryClass.IMMUTABLE_WEIGHTS, fp.Granularity.ORGAN,
            weight_bytes, home_hint=DECLARED_FPGA_HBM,
            consumers=(DECLARED_FPGA_HBM,),
        ),
        fp.SemanticObject(
            "kv_state", MemoryClass.KV_STATE, fp.Granularity.LAYER_GROUP,
            1 << 16, home_hint=HOST_GPU_UMA,
            consumers=(HOST_GPU_UMA,),
        ),
    ]
    storage, object_homes = lift_object_placements(topo, objects)

    placements = storage + (
        Placement(
            node_id="ffn_decode",
            node_kind=NodeKind.COMPUTATION,
            domain=DECLARED_FPGA_HBM,
            resources=ResourceRequirements(bytes=activation_bytes, compute_slots=1),
            validity=ValidityCondition(reason="decode-compute on declared FPGA_HBM"),
            profile_hint="INTERACTIVE",
        ),
        Placement(
            node_id="kv_update",
            node_kind=NodeKind.STATE,
            domain=HOST_GPU_UMA,
            resources=ResourceRequirements(bytes=1 << 16),
            validity=ValidityCondition(reason="KV owner stays on host GPU_UMA"),
            owner_domain=HOST_GPU_UMA,
        ),
    )

    act_cost = edge_cost_on(
        topo, HOST_GPU_UMA, DECLARED_FPGA_HBM, activation_bytes,
        note="activation host GPU_UMA -> declared FPGA_HBM; COST_MODEL",
    )
    edges = (
        SemanticTransportEdge(
            source=HOST_GPU_UMA,
            destination=DECLARED_FPGA_HBM,
            payload_semantics=PayloadSemantics.ACTIVATION,
            ownership_transfer=OwnershipTransfer.KEEP,
            sync_requirement=SyncRequirement.FENCE,
            coherency_assumption=CoherencyAssumption.NONE,
            link_coherency=CoherencyAssumption.NONE,
            cost=act_cost,
            in_transit_transforms=("pack",),
            object_id="activations",
            organ_id="ffn",
        ),
    )
    return compose(
        domains, placements, edges,
        object_placements=object_homes,
        notes=(
            "link is explicitly non-coherent; consumer waits on FENCE",
            "FPGA_HBM domain is declared, not present; costs are COST_MODEL",
            "readback and device-compute visibility are independent axes",
        ),
    )


def overclaiming_edge(base: SemanticTransportEdge,
                      assumption: CoherencyAssumption) -> SemanticTransportEdge:
    """Test helper: same edge with a (usually illegal) coherency assumption."""
    return SemanticTransportEdge(
        source=base.source,
        destination=base.destination,
        payload_semantics=base.payload_semantics,
        ownership_transfer=base.ownership_transfer,
        sync_requirement=base.sync_requirement,
        coherency_assumption=assumption,
        link_coherency=base.link_coherency,
        cost=base.cost,
        in_transit_transforms=base.in_transit_transforms,
        object_id=base.object_id,
        organ_id=base.organ_id,
        token_id=base.token_id,
        representation_id=base.representation_id,
    )


# --------------------------------------------------------------------------- fusion_isa connection


_OWNERSHIP_TO_ACQUIRE = {
    OwnershipTransfer.KEEP: FusionOp.ACQUIRE_READ,
    OwnershipTransfer.SHARE_READ: FusionOp.ACQUIRE_READ,
    OwnershipTransfer.TRANSFER: FusionOp.ACQUIRE_WRITE,
}


def plan_to_timeline(plan: HeterogeneousPlan) -> FusionTimeline:
    """Connect: HeterogeneousPlan -> fusion_isa.FusionTimeline.

    One object named by the edge.object_id (or a synthetic id). Sequence:
    ACQUIRE on the destination replica, COPY of the object onto it, FENCE
    if the edge asked for one, SUBMIT of each COMPUTATION placement,
    RELEASE. Matching fusion_isa arity: COPY has 1 in, 1 out, 1 replica;
    SUBMIT may have empty in/out/replicas; FENCE is empty.
    """
    tl = FusionTimeline()
    seq = 0
    for edge in plan.edges:
        obj = edge.object_id or f"{edge.source}_{edge.destination}"
        replica = edge.destination
        acquire_op = _OWNERSHIP_TO_ACQUIRE[edge.ownership_transfer]
        tl.submit(FusionCommand(
            seq=seq, op=acquire_op, inputs=(obj,), replicas=(replica,),
        ))
        acquire_seq = seq
        seq += 1
        tl.submit(FusionCommand(
            seq=seq, op=FusionOp.COPY,
            inputs=(obj,), outputs=(obj,), replicas=(replica,),
            depends_on=(acquire_seq,),
        ))
        copy_seq = seq
        seq += 1
        last = copy_seq
        if edge.sync_requirement is not SyncRequirement.NONE:
            tl.submit(FusionCommand(
                seq=seq, op=FusionOp.FENCE, depends_on=(copy_seq,),
            ))
            last = seq
            seq += 1
        tl.submit(FusionCommand(
            seq=seq, op=FusionOp.RELEASE, inputs=(obj,), replicas=(replica,),
            depends_on=(last,),
        ))
        seq += 1
    for p in plan.placements:
        if p.node_kind is NodeKind.COMPUTATION:
            tl.submit(FusionCommand(
                seq=seq, op=FusionOp.SUBMIT,
                outputs=(p.node_id,), replicas=(p.domain,),
            ))
            seq += 1
    return tl


# --------------------------------------------------------------------------- physical graph overlay (hcli is READ-only)


def overlay_physical_graph(graph: Mapping[str, Any], plan: HeterogeneousPlan) -> dict[str, Any]:
    """Copy of a PhysicalGraph dict with fusion-bridge fields filled in.
    Does not import or mutate hcli.physical_graph -- the compiler already
    produced `graph`; this only annotates a copy."""
    out = dict(graph)
    placement_map = {
        p.node_id: {"domain": p.domain, "kind": p.node_kind.value,
                    "resources": p.resources.to_dict()}
        for p in plan.placements
    }
    device_placement = dict(out.get("device_placement") or {})
    device_placement["fusion_bridge"] = {
        "schema": SCHEMA,
        "qualification": QUALIFICATION,
        "domains": {k: v.to_dict() for k, v in plan.domains.items()},
        "selected": {p.node_id: p.domain for p in plan.placements},
        "nodes": placement_map,
        "cost": plan.total_cost().to_dict(),
    }
    out["device_placement"] = device_placement

    sync = list(out.get("synchronization") or [])
    for edge in plan.edges:
        sync.append({
            "kind": "semantic_transport",
            "source": edge.source,
            "destination": edge.destination,
            "sync_requirement": edge.sync_requirement.value,
            "link_coherency": edge.link_coherency.name,
            "coherency_assumption": edge.coherency_assumption.name,
            "status": "planned",
        })
    out["synchronization"] = sync

    deps = list(out.get("dependencies") or [])
    for edge in plan.edges:
        deps.append({
            "from": edge.source,
            "to": edge.destination,
            "kind": "semantic_transport",
            "payload_semantics": edge.payload_semantics.value,
            "object_id": edge.object_id,
        })
    out["dependencies"] = deps
    out["fusion_bridge"] = plan.to_dict()
    return out


# --------------------------------------------------------------------------- HWIR lowering (FPGA_HBM domain)


_HWIR_TRANSFORMS = frozenset({
    "identity", "quantize", "transpose", "reduce", "checksum_digest", "pack", "unpack",
})


def _hwir_transform(name: str) -> str:
    return name if name in _HWIR_TRANSFORMS else "identity"


def _payload_to_frame(sem: PayloadSemantics) -> str:
    mapping = {
        PayloadSemantics.ACTIVATION: "activation",
        PayloadSemantics.PARTIAL_REDUCTION: "partial_reduction",
        PayloadSemantics.COMPACT_REPRESENTATION: "compact_representation_fragment",
        PayloadSemantics.STATE: "state",
        PayloadSemantics.CODEBOOK_ID: "codebook_id",
        PayloadSemantics.SPARSE_RESIDUAL: "sparse_residual",
        PayloadSemantics.WEIGHTS: "compact_representation_fragment",
        PayloadSemantics.TOKEN: "activation",
        PayloadSemantics.COMMAND: "activation",
    }
    return mapping[sem]


def _node_kind_to_hwir(kind: NodeKind) -> str:
    return {
        NodeKind.COMPUTATION: "compute",
        NodeKind.STATE: "state",
        NodeKind.STORAGE: "memory",
    }[kind]


def lower_fpga_domain_to_hwir(plan: HeterogeneousPlan, domain_name: str = DECLARED_FPGA_HBM):
    """Connect: FPGA_HBM placements + inbound edges -> tools.future.hwir graph.

    dma-transport nodes carry primitive=SemanticTransportEdge, which is
    already how hwir.PRIMITIVE_TO_NODE_KIND maps that atlas name. This is
    the same primitive, not a second one.
    """
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.future import hwir  # noqa: WPS433 -- lazy so validate() does not need it

    if domain_name not in plan.domains:
        raise FusionBridgeError(f"{domain_name!r} is not in this plan")
    if plan.domains[domain_name].kind is not DomainKind.FPGA_HBM:
        raise FusionBridgeError(
            f"{domain_name!r} is {plan.domains[domain_name].kind.value}, not FPGA_HBM")

    nodes: list[Any] = []
    edges: list[Any] = []
    placed = [p for p in plan.placements if p.domain == domain_name]
    if not placed:
        raise FusionBridgeError(f"no placements on {domain_name!r} to lower")

    for p in placed:
        inputs: dict[str, str] = {}
        outputs: dict[str, str] = {}
        if p.node_kind is NodeKind.COMPUTATION:
            inputs = {"in": "activation"}
            outputs = {"out": "activation"}
        elif p.node_kind is NodeKind.STORAGE:
            outputs = {"out": "compact_representation_fragment"}
        elif p.node_kind is NodeKind.STATE:
            inputs = {"in": "state"}
            outputs = {"out": "state"}
        nodes.append(hwir.HwirNode(
            id=p.node_id,
            kind=_node_kind_to_hwir(p.node_kind),
            primitive="FusedDecodeCompute" if p.node_kind is NodeKind.COMPUTATION
            else "StationaryRepresentation" if p.node_kind is NodeKind.STORAGE
            else "LocalStateMachine",
            organ=p.node_id,
            owner=p.owner if p.node_kind is NodeKind.STATE else None,
            inputs=inputs,
            outputs=outputs,
            transport_policy="non_coherent_inbound",
        ))

    inbound = [e for e in plan.edges if e.destination == domain_name]
    consumers = [p for p in placed if p.node_kind is NodeKind.COMPUTATION]
    consumer = consumers[0] if consumers else placed[0]
    for i, edge in enumerate(inbound):
        dma_id = f"dma_{edge.source}_{edge.destination}_{i}"
        frame = _payload_to_frame(edge.payload_semantics)
        xform = _hwir_transform(edge.in_transit_transforms[0] if edge.in_transit_transforms else "identity")
        # pack: activation -> compact_representation_fragment. Keep identity
        # when the consumer port is activation so HWIR type-check passes.
        if xform == "pack" and consumer.node_kind is NodeKind.COMPUTATION:
            xform = "identity"
        nodes.append(hwir.HwirNode(
            id=dma_id,
            kind=HWIR_NODE_KIND,
            primitive=ATLAS_PRIMITIVE,
            organ=edge.organ_id or edge.object_id,
            outputs={"out": frame},
            transport_policy=edge.link_coherency.name,
        ))
        # Consumer must accept this frame.
        for n in nodes:
            if n.id == consumer.node_id:
                n.inputs["in"] = frame
                break
        edges.append(hwir.HwirEdge(
            id=f"e_{dma_id}_{consumer.node_id}",
            src=dma_id,
            dst=consumer.node_id,
            src_port="out",
            dst_port="in",
            frame_kind=frame,
            in_transit_transform=xform,
        ))

    budget = hwir.DeviceBudget(
        device_id="declared-fpga-hbm",
        declared_not_measured=True,
        BRAM=1 << 20,
        DSP=1 << 12,
        LUT=1 << 16,
        URAM=1 << 10,
        hbm_channels=1,
    )
    graph = hwir.HwirGraph(
        model="fusion-bridge-heterogeneous",
        organ="ffn_decode",
        qualification="STATIC_ONLY",
        semantics_consumed="fusion_bridge_plan",
        nodes=nodes,
        edges=edges,
        device_budget=budget,
        notes=[
            "lowered from fusion_bridge HeterogeneousPlan",
            "declared FPGA_HBM domain; not a board, bitstream, or timing claim",
            f"atlas primitive {ATLAS_PRIMITIVE} -> {HWIR_NODE_KIND}",
        ],
    )
    return graph


# --------------------------------------------------------------------------- physical-graph compile helper used by the odyssey adapter


def compile_and_overlay(architecture: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Connect: hcli.physical_graph.compile_physical_graph -> overlay.
    Architecture default is empty organs; this is a planning annotation,
    not a performance claim."""
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from hcli.physical_graph import compile_physical_graph  # noqa: WPS433

    graph = compile_physical_graph(architecture or {"model_id": "fusion-bridge", "organs": []})
    plan = two_domain_plan()
    return overlay_physical_graph(graph, plan)


def main() -> int:
    plan = two_domain_plan()
    report = validate(plan)
    if not report.ok:
        for e in report.errors:
            print(f"{e['code']}: {e['path']}: {e['message']}")
        return 1
    cost = plan.total_cost()
    print(f"ok domains={sorted(plan.domains)} "
          f"placements={len(plan.placements)} edges={len(plan.edges)} "
          f"cost_label={cost.label} qualification={plan.qualification}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
