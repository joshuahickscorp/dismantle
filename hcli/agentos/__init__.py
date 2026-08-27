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
    raise AttributeError(name)
