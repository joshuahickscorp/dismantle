"""FPGA pre-board contracts and model-specific HWIR maps.

This module provides compiler-facing scaffolding for both Qwen systems.  The
only execution surface is a deterministic mock/link simulator; all simulated
numbers carry ``[S]`` and no physical board or U50 performance is claimed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.fpga_preboard.v1"
VERIFIED = "[V]"
DERIVED = "[D]"
SIMULATED = "[S]"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class FPGADeviceGenome:
    device_id: str = "unselected-fpga-device"
    vendor: str = "unselected"
    family: str = "unselected"
    status: str = "[D] TARGET_UNSELECTED"
    hbm_channels: int = 0
    pcie_generation: str = "unselected"
    physical_board_present: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": "hcli.fpga.device_genome.v1", "label": DERIVED, **self.__dict__}


@dataclass(frozen=True)
class HBMGenome:
    channels: int = 0
    capacity_bytes: Optional[int] = None
    bandwidth_gbps: Optional[float] = None
    status: str = "[D] TARGET_UNSELECTED"

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": "hcli.fpga.hbm_genome.v1", "label": DERIVED, **self.__dict__}


@dataclass
class HWIR:
    model: str
    nodes: list[Dict[str, Any]] = field(default_factory=list)
    buffers: list[Dict[str, Any]] = field(default_factory=list)
    dependencies: list[Dict[str, Any]] = field(default_factory=list)
    synchronization: list[Dict[str, Any]] = field(default_factory=list)
    placements: list[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema": "hcli.fpga.hwir.v1",
            "label": DERIVED,
            "model": self.model,
            "nodes": self.nodes,
            "buffers": self.buffers,
            "dependencies": self.dependencies,
            "synchronization": self.synchronization,
            "placements": self.placements,
        }
        body["fingerprint"] = _hash(body)
        return body


class FPGAProvider:
    """Stable provider contract; concrete execution is intentionally separate."""

    def identity(self) -> Dict[str, Any]:
        return {"provider": "fpga", "status": "UNIMPLEMENTED", "physical_board": False}

    def capabilities(self) -> Dict[str, Any]:
        return {"hwir": True, "simulation": False, "physical_execution": False}

    def health(self) -> Dict[str, Any]:
        return {"status": "NOT_AVAILABLE", "physical_board": False}

    def execute(self, hwir: Mapping[str, Any]) -> Dict[str, Any]:
        del hwir
        raise RuntimeError("FPGAProvider has no physical backend selected")


class MockFPGAProvider(FPGAProvider):
    """Schema/link-simulation provider; it cannot return hardware evidence."""

    def identity(self) -> Dict[str, Any]:
        return {"provider": "mock-fpga", "status": "SIMULATOR_ONLY", "physical_board": False, "label": SIMULATED}

    def capabilities(self) -> Dict[str, Any]:
        return {"hwir": True, "link_simulation": True, "cycle_simulation": False, "physical_execution": False}

    def health(self) -> Dict[str, Any]:
        return {"status": "AVAILABLE_FOR_SCHEMA_SIMULATION", "physical_board": False}

    def execute(self, hwir: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "schema": "hcli.fpga.receipt.v1",
            "status": "SIMULATED",
            "label": SIMULATED,
            "hwir_fingerprint": hwir.get("fingerprint"),
            "physical_execution": False,
            "performance_claim": False,
            "claim_boundary": "mock provider executes no FPGA hardware",
        }


class TransportLinkSimulator:
    """Simple transparent link model used only for sensitivity planning."""

    def __init__(self, *, bandwidth_gbps: float = 64.0, latency_ns: float = 900.0) -> None:
        self.bandwidth_gbps = float(bandwidth_gbps)
        self.latency_ns = float(latency_ns)

    def transfer(self, bytes_count: int, *, hops: int = 1) -> Dict[str, Any]:
        payload = max(0, int(bytes_count))
        seconds = payload / max(1.0, self.bandwidth_gbps * 1_000_000_000 / 8)
        return {
            "label": SIMULATED,
            "bytes": payload,
            "hops": max(1, int(hops)),
            "bandwidth_gbps": self.bandwidth_gbps,
            "latency_ns": self.latency_ns,
            "estimated_transfer_ns": (self.latency_ns + seconds * 1_000_000_000) * max(1, int(hops)),
            "status": "SENSITIVITY_ONLY",
        }


def partitioner(model: str, organs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Produce a candidate within-organ partition map, never a hardware claim."""
    model_key = str(model)
    rows = []
    for organ in organs:
        name = str(organ.get("organ") or organ.get("id") or "unknown")
        if model_key == "qwen27":
            device = "APPLE_UMA_PLUS_FPGA_HBM_HYPOTHESIS"
            transport = "activations_and_partial_reductions_only"
        else:
            device = "FPGA_HBM_FOR_ROUTING_STATE_AND_EXPERT_SUBSETS_HYPOTHESIS"
            transport = "route_metadata_activations_and_partial_reductions_only"
        rows.append({
            "organ": name,
            "device_assignment": device,
            "partition_axis": "within_organ_tensor_parallel",
            "resident_weight_policy": "resident_shards_no_weight_body_per_token_transfer",
            "transport_policy": transport,
            "label": DERIVED,
        })
    return {"schema": "hcli.fpga.partitioner.v1", "model": model_key, "partitions": rows, "status": "CANDIDATE_ONLY"}


def _qwen27_organs() -> list[Dict[str, Any]]:
    return [
        {"organ": "mlp_gate_up_down", "mapping": "within-organ tensor parallel low-bit GEMV + epilogue", "priority": "P0"},
        {"organ": "gqa_qkv_and_output", "mapping": "projection shards with activation/partial-reduction transport", "priority": "P0"},
        {"organ": "deltanet_state_and_input_projection", "mapping": "persistent state machine with resident state", "priority": "P0"},
        {"organ": "norm_add_epilogues", "mapping": "fused epilogue near producer", "priority": "P1"},
        {"organ": "lm_head_and_sampling", "mapping": "partitioned vocabulary reduction and selection", "priority": "P1"},
        {"organ": "command_buffer_graph", "mapping": "persistent scheduling and reduced synchronization", "priority": "P1"},
    ]


def _flash_organs() -> list[Dict[str, Any]]:
    return [
        {"organ": "expert_bank", "mapping": "HBM-resident selected expert subsets with native NF GEMV", "priority": "P0"},
        {"organ": "router_topk_and_gather", "mapping": "irregular route/index processing close to expert bank", "priority": "P0"},
        {"organ": "routed_plus_shared_expert", "mapping": "fused selected-expert execution and epilogue", "priority": "P0"},
        {"organ": "deltanet_persistent_state", "mapping": "state-machine/update pipeline with resident state", "priority": "P0"},
        {"organ": "ngram_lookup_or_generator", "mapping": "HBM lookup/compositional generator rather than matrix GEMV", "priority": "P1"},
        {"organ": "sparse_attention", "mapping": "indexer budget traversal, sparse KV gather, reductions", "priority": "P1"},
        {"organ": "mtp_draft_verify_rollback", "mapping": "explicit draft/accept/reject/state rollback accounting", "priority": "P1"},
    ]


def _hwir(model: str, organs: list[Dict[str, Any]]) -> HWIR:
    nodes = [
        {"id": row["organ"], "kind": "organ_operator", "mapping": row["mapping"], "label": DERIVED}
        for row in organs
    ]
    buffers = [
        {"id": "resident_weight_shards", "lifetime": "persistent", "per_token_transfer": False, "label": DERIVED},
        {"id": "activations", "lifetime": "token", "per_token_transfer": True, "label": DERIVED},
        {"id": "partial_reductions", "lifetime": "token", "per_token_transfer": True, "label": DERIVED},
        {"id": "persistent_state", "lifetime": "sequence", "per_token_transfer": False, "label": DERIVED},
    ]
    dependencies = [{"from": organs[index - 1]["organ"], "to": row["organ"], "kind": "token_dependency", "label": DERIVED} for index, row in enumerate(organs) if index]
    synchronization = [{"boundary": row["organ"], "policy": "minimize_command_buffer_and_cross_device_sync", "label": DERIVED} for row in organs]
    placements = partitioner(model, organs)["partitions"]
    return HWIR(model=model, nodes=nodes, buffers=buffers, dependencies=dependencies, synchronization=synchronization, placements=placements)


def _map(model: str) -> Dict[str, Any]:
    organs = _qwen27_organs() if model == "qwen27" else _flash_organs()
    hwir = _hwir(model, organs).to_dict()
    simulator = TransportLinkSimulator()
    link_rows = [simulator.transfer(size, hops=hops) for size, hops in ((1 << 20, 1), (16 << 20, 1), (64 << 20, 2))]
    device = FPGADeviceGenome()
    hbm = HBMGenome()
    provider = MockFPGAProvider()
    experiment_dag = {
        "schema": "hcli.fpga.experiment_dag.v1",
        "nodes": [
            {"id": "metadata_identity", "dependencies": [], "status": "READY", "label": DERIVED},
            {"id": "hwir_compile", "dependencies": ["metadata_identity"], "status": "READY", "label": DERIVED},
            {"id": "link_sensitivity", "dependencies": ["hwir_compile"], "status": "SIMULATED", "label": SIMULATED},
            {"id": "native_parity", "dependencies": ["hwir_compile"], "status": "WAITING_FOR_PHYSICAL_OR_NATIVE_BACKEND", "label": DERIVED},
            {"id": "hardware_receipt", "dependencies": ["native_parity", "link_sensitivity"], "status": "BLOCKED_NO_BOARD", "label": DERIVED},
        ],
    }
    cache_material = {"hwir": hwir, "device": device.to_dict(), "hbm": hbm.to_dict(), "toolchain": ["vivado", "vitis", "v++", "xsim"]}
    return {
        "schema": "hcli.fpga.organ_map.v1",
        "model": model,
        "label": DERIVED,
        "model_identity": (
            {"label": "Qwen3.8-27B sealed resident / NOETIC_PARENT_A", "profile": "hcli/hawking-native.sealed-3.14.json", "artifact_identity": "NOETIC_PARENT_A"}
            if model == "qwen27" else
            {"label": REPO_ID, "pinned_revision": PINNED_REVISION, "source_identity": "pinned metadata; weights/native executable absent"}
        ),
        "organs": organs,
        "device_genome": device.to_dict(),
        "hbm_genome": hbm.to_dict(),
        "hwir": hwir,
        "partitioner": partitioner(model, organs),
        "transport_link_simulator": {
            "schema": "hcli.fpga.link.v1",
            "status": "SIMULATION_ONLY",
            "parameters": {"bandwidth_gbps": simulator.bandwidth_gbps, "latency_ns": simulator.latency_ns},
            "rows": link_rows,
            "label": SIMULATED,
        },
        "experiment_dag": experiment_dag,
        "module_cache": {
            "schema": "hcli.fpga.module_cache.v1",
            "key_algorithm": "sha256(canonical HWIR + device genome + HBM genome + toolchain identity)",
            "candidate_key": _hash(cache_material),
            "status": "SCHEMA_ONLY",
            "label": DERIVED,
        },
        "reproducible_vivado_vitis_harness": {
            "schema": "hcli.fpga.harness.v1",
            "status": "CONTRACT_ONLY",
            "commands": [
                ["vivado", "-mode", "batch", "-source", "build.tcl"],
                ["vitis", "--workspace", "workspace"],
                ["v++", "--link", "--config", "link.cfg"],
                ["xsim", "hwir_tb"],
            ],
            "inputs": ["HWIR fingerprint", "kernel genome", "target/device identity", "reference vectors"],
            "label": DERIVED,
        },
        "driverkit_bridge_design": {
            "status": "DESIGN_NOT_IMPLEMENTED",
            "notes": ["user-space bridge owns explicit device identity", "DMA buffers are bounded and receipt-addressable", "no kernel extension or hardware permission is claimed"],
            "label": DERIVED,
        },
        "rtl_hls_verifier_surface": {
            "status": "CONTRACT_ONLY",
            "checks": ["bitstream/module hash", "HWIR fingerprint", "input/reference parity", "resource bounds", "transport trace", "capability contract"],
            "label": DERIVED,
        },
        "provider": provider.identity(),
        "provider_capabilities": provider.capabilities(),
        "simulation_receipt": provider.execute(hwir),
        "claim_boundary": "Candidate FPGA mapping only. [D] denotes a derived hypothesis and [S] denotes link/schema simulation. No physical board, bitstream, hardware timing, or U50 performance is claimed.",
    }


def run_fpga_preboard(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    qwen27 = _map("qwen27")
    flash = _map("flash-next")
    qwen_path = repo / "receipts" / "headless" / "QWEN27_FPGA_ORGAN_MAP.json"
    flash_path = repo / "receipts" / "headless" / "FLASH_NEXT_FPGA_ORGAN_MAP.json"
    atomic_write_json(qwen_path, qwen27)
    atomic_write_json(flash_path, flash)
    report = {
        "schema": SCHEMA,
        "status": "PASSED",
        "generated_at": time.time(),
        "repo_root": str(repo),
        "maps": {"qwen27": str(qwen_path), "flash_next": str(flash_path)},
        "shared_primitives": ["packed low-bit GEMV", "norm/epilogue", "persistent state/update", "transport/link accounting", "kernel genome/cache", "telemetry/receipt verifier"],
        "model_specific": {"qwen27": [row["organ"] for row in qwen27["organs"]], "flash_next": [row["organ"] for row in flash["organs"]]},
        "physical_board": {"status": "ABSENT", "claim": False},
        "fpga_backend": {"status": "NOT_BUILT", "claim": False},
        "simulation": {"qwen27": "[S] link sensitivity only", "flash_next": "[S] link sensitivity only"},
        "fingerprint": _hash({"qwen27": qwen27["hwir"], "flash_next": flash["hwir"]}),
        "checks": {
            "both_model_maps_written": qwen_path.is_file() and flash_path.is_file(),
            "qwen27_hwir_present": bool(qwen27.get("hwir", {}).get("fingerprint")),
            "flash_next_hwir_present": bool(flash.get("hwir", {}).get("fingerprint")),
            "mock_provider_is_simulation_only": qwen27.get("simulation_receipt", {}).get("physical_execution") is False and flash.get("simulation_receipt", {}).get("physical_execution") is False,
            "no_physical_board_claim": qwen27["device_genome"]["physical_board_present"] is False and flash["device_genome"]["physical_board_present"] is False,
        },
        "claim_boundary": "Pre-board compiler scaffolding for both Qwen systems. No physical FPGA execution or performance is claimed.",
    }
    report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / "HCLI_FPGA_PREBOARD.json"
    atomic_write_json(destination, report)
    report["receipt_path"] = str(destination)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_fpga_preboard(repo_root=args.repo_root, emit=args.emit)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DERIVED", "FPGADeviceGenome", "FPGAProvider", "HBMGenome", "HWIR", "MockFPGAProvider", "SCHEMA", "SIMULATED", "TransportLinkSimulator", "main", "partitioner", "run_fpga_preboard"]


if __name__ == "__main__":
    raise SystemExit(main())
