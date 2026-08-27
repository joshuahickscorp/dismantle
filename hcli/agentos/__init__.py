"""AgentOS: Goal, obligations, WorkUnit DAG, scheduler, repair,
mutation authority, verifier orchestration, mission persistence, steering.

Canonical implementation modules remain ``hcli.goal``, ``hcli.workunit``,
``hcli.scheduler``, ``hcli.mission``, ``hcli.ledger``, ``hcli.steering``,
``hcli.mutation``, ``hcli.verifier_pipeline``, ``hcli.dag_store``,
``hcli.executors``, ``hcli.resources``. Re-exported here so ownership is
an importable package, not a comment — the class objects are the same.
"""
from hcli.dag_store import DagStore
from hcli.executors import WorkUnitExecutor
from hcli.goal import GoalCompiler, WorkerPacket, compile_worker_context
from hcli.ledger import Ledger
from hcli.mission import Mission
from hcli.mutation import MutationError
from hcli.resources import MutationLock, ResourceClass, ResourceLimits
from hcli.scheduler import Scheduler
from hcli.steering import SteeringQueue
from hcli.verifier_pipeline import command_is_admissible
from hcli.workunit import WorkUnit
from hcli.agentos.runtime import AgentOS
from hcli.agentos.background import BackgroundJob, BackgroundJobStore
from hcli.agentos.states import AgentState, mission_state, workunit_state
from hcli.providers import (
    Capability,
    CapabilityContract,
    GenerationRequest,
    GenerationResponse,
    ModelProvider,
    ProviderFailure,
    ProviderHealth,
    ProviderReceipt,
    ResidentProfile,
    ResidentProvider,
    RolePolicy,
    RoleRouter,
    RuntimeGenome,
)
from hcli.physical_graph import PhysicalGraph, compile_physical_graph
from hcli.result_envelope import ResultEnvelope, build_result_envelope
from hcli.tool_registry import ToolContext, ToolRegistry, ToolResult, ToolSpec, default_tool_registry

__all__ = [
    "DagStore",
    "GoalCompiler",
    "Ledger",
    "Mission",
    "MutationError",
    "MutationLock",
    "ResourceClass",
    "ResourceLimits",
    "Scheduler",
    "SteeringQueue",
    "WorkUnit",
    "WorkUnitExecutor",
    "WorkerPacket",
    "command_is_admissible",
    "compile_worker_context",
    "AgentOS",
    "BackgroundJob",
    "BackgroundJobStore",
    "RECOVERY_GATE_SCHEMA",
    "run_recovery_gate",
    "RESEARCH_GATE_SCHEMA",
    "run_research_gate",
    "VMCP_GATE_SCHEMA",
    "run_vmcp_gate",
    "NATIVE_GATE_SCHEMA",
    "run_native_gate",
    "RESIDENT_GATE_SCHEMA",
    "run_resident_gate",
    "NATIVE_MISSION_GATE_SCHEMA",
    "run_native_mission_gate",
    "AUTONOMY_GATE_SCHEMA",
    "run_autonomy_gate",
    "UNATTENDED_WINDOW_SCHEMA",
    "run_unattended_window",
    "ACCELERATOR_REGRESSION_SCHEMA",
    "run_accelerator_regression",
    "MODELLAKE_CENSUS_SCHEMA",
    "run_modellake_census",
    "FLASH_SCIENCE_SCHEMA",
    "run_flash_science_gate",
    "PREBOARD_SCHEMA",
    "run_preboard",
    "INITIAL_CHARGE_SCHEMA",
    "create_initial_charge",
    "TRANSFER_MAP_SCHEMA",
    "PRECEDENT_MAP_SCHEMA",
    "write_science_maps",
    "DENSE_NF_AB_SCHEMA",
    "evaluate_ab",
    "run_ab_scaffold",
    "FPGA_PREBOARD_SCHEMA",
    "run_fpga_preboard",
    "MODELLAKE_SUPERVISION_SCHEMA",
    "run_model_lake_supervision",
    "OVERNIGHT_HANDOFF_SCHEMA",
    "build_handoff",
    "AgentState",
    "mission_state",
    "workunit_state",
    "Capability",
    "CapabilityContract",
    "GenerationRequest",
    "GenerationResponse",
    "ModelProvider",
    "ProviderFailure",
    "ProviderHealth",
    "ProviderReceipt",
    "ResidentProfile",
    "ResidentProvider",
    "RolePolicy",
    "RoleRouter",
    "RuntimeGenome",
    "PhysicalGraph",
    "compile_physical_graph",
    "ResultEnvelope",
    "build_result_envelope",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "default_tool_registry",
]


def __getattr__(name: str):
    """Load executable operational gates lazily so ``python -m`` is clean."""
    if name in {"RECOVERY_GATE_SCHEMA", "run_recovery_gate"}:
        from hcli.agentos import recovery

        return recovery.SCHEMA if name == "RECOVERY_GATE_SCHEMA" else recovery.run_recovery_gate
    if name in {"RESEARCH_GATE_SCHEMA", "run_research_gate"}:
        from hcli.agentos import research

        return research.SCHEMA if name == "RESEARCH_GATE_SCHEMA" else research.run_research_gate
    if name in {"VMCP_GATE_SCHEMA", "run_vmcp_gate"}:
        from hcli.agentos import vmcp_gate

        return vmcp_gate.SCHEMA if name == "VMCP_GATE_SCHEMA" else vmcp_gate.run_vmcp_gate
    if name in {"NATIVE_GATE_SCHEMA", "run_native_gate"}:
        from hcli.agentos import native_gate

        return native_gate.SCHEMA if name == "NATIVE_GATE_SCHEMA" else native_gate.run_native_gate
    if name in {"RESIDENT_GATE_SCHEMA", "run_resident_gate"}:
        from hcli.agentos import resident_gate

        return resident_gate.SCHEMA if name == "RESIDENT_GATE_SCHEMA" else resident_gate.run_resident_gate
    if name in {"NATIVE_MISSION_GATE_SCHEMA", "run_native_mission_gate"}:
        from hcli.agentos import native_mission_gate

        return native_mission_gate.SCHEMA if name == "NATIVE_MISSION_GATE_SCHEMA" else native_mission_gate.run_native_mission_gate
    if name in {"AUTONOMY_GATE_SCHEMA", "run_autonomy_gate", "UNATTENDED_WINDOW_SCHEMA", "run_unattended_window"}:
        from hcli.agentos import autonomy_gate

        if name == "AUTONOMY_GATE_SCHEMA":
            return autonomy_gate.SCHEMA
        if name == "UNATTENDED_WINDOW_SCHEMA":
            return autonomy_gate.WINDOW_SCHEMA
        return autonomy_gate.run_autonomy_gate if name == "run_autonomy_gate" else autonomy_gate.run_unattended_window
    if name in {"ACCELERATOR_REGRESSION_SCHEMA", "run_accelerator_regression"}:
        from hcli.agentos import accelerator_regression

        return accelerator_regression.SCHEMA if name == "ACCELERATOR_REGRESSION_SCHEMA" else accelerator_regression.run_accelerator_regression
    if name in {"MODELLAKE_CENSUS_SCHEMA", "run_modellake_census"}:
        from hcli.agentos import modellake_gate

        return modellake_gate.SCHEMA if name == "MODELLAKE_CENSUS_SCHEMA" else modellake_gate.run_modellake_census
    if name in {"FLASH_SCIENCE_SCHEMA", "run_flash_science_gate"}:
        from hcli.agentos import flash_science

        return flash_science.SCHEMA if name == "FLASH_SCIENCE_SCHEMA" else flash_science.run_flash_science_gate
    if name in {"PREBOARD_SCHEMA", "run_preboard"}:
        from hcli.agentos import preboard

        return preboard.SCHEMA if name == "PREBOARD_SCHEMA" else preboard.run_preboard
    if name in {"INITIAL_CHARGE_SCHEMA", "create_initial_charge"}:
        from hcli.agentos import charge

        return charge.SCHEMA if name == "INITIAL_CHARGE_SCHEMA" else charge.create_initial_charge
    if name in {"TRANSFER_MAP_SCHEMA", "PRECEDENT_MAP_SCHEMA", "write_science_maps"}:
        from hcli.agentos import science_maps

        if name == "TRANSFER_MAP_SCHEMA":
            return science_maps.TRANSFER_SCHEMA
        if name == "PRECEDENT_MAP_SCHEMA":
            return science_maps.PRECEDENT_SCHEMA
        return science_maps.write_science_maps
    if name in {"DENSE_NF_AB_SCHEMA", "evaluate_ab", "run_ab_scaffold"}:
        from hcli.agentos import representation_ab

        if name == "DENSE_NF_AB_SCHEMA":
            return representation_ab.SCHEMA
        if name == "evaluate_ab":
            return representation_ab.evaluate_ab
        return representation_ab.run_ab_scaffold
    if name in {"FPGA_PREBOARD_SCHEMA", "run_fpga_preboard"}:
        from hcli.agentos import fpga_preboard

        return fpga_preboard.SCHEMA if name == "FPGA_PREBOARD_SCHEMA" else fpga_preboard.run_fpga_preboard
    if name in {"MODELLAKE_SUPERVISION_SCHEMA", "run_model_lake_supervision"}:
        from hcli.agentos import modellake_supervisor

        return modellake_supervisor.SCHEMA if name == "MODELLAKE_SUPERVISION_SCHEMA" else modellake_supervisor.run_model_lake_supervision
    if name in {"OVERNIGHT_HANDOFF_SCHEMA", "build_handoff"}:
        from hcli.agentos import handoff

        return handoff.SCHEMA if name == "OVERNIGHT_HANDOFF_SCHEMA" else handoff.build_handoff
    raise AttributeError(name)
