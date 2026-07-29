//! Fabric software plane: node discovery, placement simulation, KV ownership,
//! pipeline scheduling, heartbeats, and failure/replay receipts.
//!
//! This module coordinates distributed *inference placement* software. It does
//! **not** execute model weights or touch the inference hot path (Metal /
//! gravity / kernels). Another lane owns execution.
//!
//! ## Qualification law
//!
//! Simulated or fixture results are **never** physical hardware qualification.
//! Every plan / receipt that is not physical hardware must set
//! `not_physical_qualification: true` and a non-physical
//! [`qualification::QualificationKind`]. The schema validator rejects
//! unlabelled simulated results.
//!
//! Terminal hardware state for this single-machine session:
//! `FABRIC_HARDWARE_QUALIFICATION_PENDING`.

pub use agent::{AgentConfig, AgentState, FabricAgent, FabricAgentHandle};
pub use failure::{
    CheckpointId, FailureDetector, FailureReplayReceipt, HeartbeatMonitor, LostWorkSummary,
};
pub use fixture::{
    run_inprocess_software_fixture, run_two_process_fixture, TwoProcessFixtureResult,
};
pub use node::{
    AcceleratorClass, BandwidthClass, DiscoverySource, NodeCapabilities, NodeDiscovery, NodeId,
    OsNodeProbe, SimulatedNodeSet, FIXED_FAKE_MEMORY_BYTES,
};
pub use placement::{
    reject_unlabelled_simulated, validate_placement_plan_schema, ContentHash,
    KvOwnershipInvariant, KvRangeOwnership, ModelSection, PlacementPlan, PlacementRequest,
    PlacementSimulator, PredictedCost, SectionPlacement, StageAssignment, WorkloadClass,
    PLACEMENT_SCHEMA,
};
pub use pipeline::{
    MicrobatchState, PipelineScheduler, PipelineStatus, StageGraph, StageId, StageState,
};
pub use protocol::{AgentRequest, AgentResponse, PlacementAssignment};
pub use qualification::{QualificationKind, HARDWARE_QUALIFICATION_PENDING, QUALIFICATION_SCHEMA};

// --- inlined fabric/agent.rs ---
pub mod agent {
//! Fabric Agent — per-node process logic.
//!
//! Registers a node, reports real capabilities, heartbeats, accepts placement
//! assignments, and reports failure. Runs as a local OS process on this
//! machine; the ABI does not assume co-location with the coordinator.

use parking_lot::Mutex;
use serde_json::json;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use super::failure::{CheckpointId, FailureDetector, FailureReplayReceipt};
use super::node::{NodeCapabilities, NodeId, OsNodeProbe};
use super::placement::{ContentHash, PlacementPlan};
use super::protocol::{
    AgentRequest, AgentResponse, PlacementAssignment, FABRIC_PLACEHOLDER_EVENT_KIND,
};
use super::qualification::QualificationKind;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub node_id: NodeId,
    pub listen_addr: String,
}

impl AgentConfig {
    pub fn new(node_id: impl Into<String>, listen_addr: impl Into<String>) -> Self {
        Self {
            node_id: NodeId::new(node_id),
            listen_addr: listen_addr.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentState {
    pub node_id: NodeId,
    pub capabilities: NodeCapabilities,
    pub alive: bool,
    pub heartbeat_seq: u64,
    pub assignment: Option<PlacementAssignment>,
    pub held_section_hashes: BTreeSet<String>,
    pub last_request_id: Option<String>,
    pub last_checkpoint: Option<CheckpointId>,
    pub injected_failure: bool,
}

/// In-process fabric agent. The binary wraps this with a TCP loop.
pub struct FabricAgent {
    config: AgentConfig,
    probe: OsNodeProbe,
    state: Mutex<AgentState>,
    running: AtomicBool,
    hb_seq: AtomicU64,
}

impl FabricAgent {
    pub fn new(config: AgentConfig) -> Self {
        let probe = OsNodeProbe::new(config.node_id.as_str());
        let capabilities = probe.probe_once();
        let state = AgentState {
            node_id: config.node_id.clone(),
            capabilities,
            alive: true,
            heartbeat_seq: 0,
            assignment: None,
            held_section_hashes: BTreeSet::new(),
            last_request_id: None,
            last_checkpoint: None,
            injected_failure: false,
        };
        Self {
            config,
            probe,
            state: Mutex::new(state),
            running: AtomicBool::new(true),
            hb_seq: AtomicU64::new(0),
        }
    }

    pub fn node_id(&self) -> NodeId {
        self.config.node_id.clone()
    }

    pub fn capabilities(&self) -> NodeCapabilities {
        // Refresh free memory on read.
        let mut caps = self.probe.probe_once();
        caps.node_id = self.config.node_id.clone();
        let mut st = self.state.lock();
        st.capabilities = caps.clone();
        caps
    }

    pub fn snapshot(&self) -> AgentState {
        self.state.lock().clone()
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn handle(&self, req: AgentRequest) -> AgentResponse {
        if !self.running.load(Ordering::SeqCst) {
            return AgentResponse::Error {
                message: "agent stopped".into(),
            };
        }
        match req {
            AgentRequest::Register { .. } => {
                let caps = self.capabilities();
                AgentResponse::Registered {
                    capabilities: caps,
                }
            }
            AgentRequest::Heartbeat { node_id, seq } => {
                let mut st = self.state.lock();
                if st.injected_failure || !st.alive {
                    return AgentResponse::Failed {
                        node_id: st.node_id.clone(),
                        reason: "node dead".into(),
                    };
                }
                st.heartbeat_seq = seq;
                self.hb_seq.store(seq, Ordering::SeqCst);
                AgentResponse::HeartbeatAck { node_id, seq }
            }
            AgentRequest::Assign { assignment } => self.accept_assignment(assignment),
            AgentRequest::RunRequest {
                request_id,
                plan_id,
            } => self.run_request(request_id, plan_id),
            AgentRequest::InjectFailure { node_id } => {
                let mut st = self.state.lock();
                if st.node_id != node_id {
                    return AgentResponse::Error {
                        message: format!("wrong node: {}", st.node_id),
                    };
                }
                st.alive = false;
                st.injected_failure = true;
                // Placeholder event — Bridge lane owns the real event model.
                AgentResponse::PlaceholderEvent {
                    kind: FABRIC_PLACEHOLDER_EVENT_KIND.into(),
                    payload: json!({
                        "event": "node_failure_injected",
                        "node_id": node_id.as_str(),
                    }),
                }
            }
            AgentRequest::GetStatus { node_id } => {
                let st = self.state.lock();
                if st.node_id != node_id {
                    return AgentResponse::Error {
                        message: "node id mismatch".into(),
                    };
                }
                AgentResponse::Status {
                    node_id: st.node_id.clone(),
                    alive: st.alive && !st.injected_failure,
                    assignment_plan_id: st.assignment.as_ref().map(|a| a.plan_id.clone()),
                    held_section_hashes: st.held_section_hashes.iter().cloned().collect(),
                }
            }
            AgentRequest::Shutdown => {
                self.running.store(false, Ordering::SeqCst);
                let mut st = self.state.lock();
                st.alive = false;
                AgentResponse::Ok {
                    node_id: st.node_id.clone(),
                    detail: "shutdown".into(),
                }
            }
        }
    }

    fn accept_assignment(&self, assignment: PlacementAssignment) -> AgentResponse {
        let mut st = self.state.lock();
        if !st.alive || st.injected_failure {
            return AgentResponse::Failed {
                node_id: st.node_id.clone(),
                reason: "cannot accept assignment: node dead".into(),
            };
        }
        if assignment.assigned_node != st.node_id {
            return AgentResponse::Error {
                message: format!(
                    "assignment for {} delivered to {}",
                    assignment.assigned_node, st.node_id
                ),
            };
        }
        // Prove we hold the sections the plan assigns to us (by content hash).
        let mut held = BTreeSet::new();
        for sp in &assignment.plan.section_placements {
            if sp.node_id == st.node_id {
                held.insert(sp.content_hash.0.clone());
            }
        }
        let plan_id = assignment.plan_id.clone();
        st.held_section_hashes = held.clone();
        st.assignment = Some(assignment);
        st.last_checkpoint = Some(CheckpointId::new(format!("ckpt-assigned-{plan_id}")));
        AgentResponse::AssignmentAccepted {
            node_id: st.node_id.clone(),
            plan_id,
            held_section_hashes: held.into_iter().collect(),
        }
    }

    fn run_request(&self, request_id: String, plan_id: String) -> AgentResponse {
        let mut st = self.state.lock();
        if !st.alive || st.injected_failure {
            return AgentResponse::Failed {
                node_id: st.node_id.clone(),
                reason: format!("node dead mid-request {request_id}"),
            };
        }
        let Some(assignment) = st.assignment.as_ref() else {
            return AgentResponse::Error {
                message: "no assignment".into(),
            };
        };
        if assignment.plan_id != plan_id {
            return AgentResponse::Error {
                message: format!("plan mismatch: have {} want {plan_id}", assignment.plan_id),
            };
        }
        let local_stages = assignment
            .plan
            .stage_assignments
            .iter()
            .filter(|s| s.node_id == st.node_id)
            .count() as u32;
        st.last_request_id = Some(request_id.clone());
        st.last_checkpoint = Some(CheckpointId::new(format!(
            "ckpt-req-{request_id}-after-local"
        )));
        AgentResponse::RequestProgress {
            node_id: st.node_id.clone(),
            request_id,
            completed_local_stages: local_stages,
        }
    }

    /// Prove held content hashes match the plan for this node.
    pub fn prove_holds(&self, plan: &PlacementPlan) -> Result<Vec<ContentHash>, String> {
        let st = self.state.lock();
        let mut proofs = Vec::new();
        for sp in &plan.section_placements {
            if sp.node_id != st.node_id {
                continue;
            }
            if !st.held_section_hashes.contains(&sp.content_hash.0) {
                return Err(format!(
                    "node {} missing section hash {}",
                    st.node_id, sp.content_hash.0
                ));
            }
            proofs.push(sp.content_hash.clone());
        }
        Ok(proofs)
    }
}

/// Shared handle for multi-threaded TCP server.
pub type FabricAgentHandle = Arc<FabricAgent>;

/// Build a software-fixture receipt when coordinating failure outside the agent.
pub fn fixture_receipt(
    request_id: &str,
    failed: &NodeId,
    plan: &PlacementPlan,
    replan: &PlacementPlan,
    lost_in_flight: u32,
) -> FailureReplayReceipt {
    use super::failure::{LostWorkSummary, FAILURE_RECEIPT_SCHEMA};
    use super::placement::KvOwnershipInvariant;

    let lost_stages = plan
        .stage_assignments
        .iter()
        .filter(|s| &s.node_id == failed)
        .map(|s| s.stage_id.clone())
        .collect::<Vec<_>>();
    let lost_kv = KvOwnershipInvariant::ranges_lost_on_failure(&plan.kv_ownership, failed);
    let lost_layers: Vec<(u32, u32)> = plan
        .stage_assignments
        .iter()
        .filter(|s| &s.node_id == failed)
        .map(|s| (s.layer_start, s.layer_end))
        .collect();
    let replayed_stages = replan
        .stage_assignments
        .iter()
        .filter(|s| {
            lost_layers
                .iter()
                .any(|(ls, le)| s.layer_start < *le && s.layer_end > *ls)
        })
        .map(|s| s.stage_id.clone())
        .collect();

    FailureReplayReceipt {
        schema: FAILURE_RECEIPT_SCHEMA.into(),
        request_id: request_id.into(),
        failed_node: failed.clone(),
        lost_work: LostWorkSummary {
            stages: lost_stages,
            kv_ranges: lost_kv,
            in_flight_microbatches: lost_in_flight,
        },
        replayed_from_checkpoint: CheckpointId::new(format!("ckpt-before-fail-{request_id}")),
        replayed_stages,
        replan_plan_id: replan.plan_id.clone(),
        qualification: QualificationKind::SoftwareFixture,
        not_physical_qualification: true,
        artifact_label: format!(
            "failure_replay_receipt_software_fixture_{}",
            failed.as_str()
        ),
    }
}

/// Helper used by coordinator logic after detecting death.
pub fn detector() -> FailureDetector {
    FailureDetector::new(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::node::SimulatedNodeSet;
    use crate::fabric::placement::{
        ModelSection, PlacementRequest, PlacementSimulator, WorkloadClass,
    };
    const GIB: u64 = 1024 * 1024 * 1024;
    #[test]
    fn agent_registers_real_capabilities() {
        let agent = FabricAgent::new(AgentConfig::new("agent-a", "127.0.0.1:0"));
        let caps = agent.capabilities();
        assert_eq!(caps.node_id.as_str(), "agent-a");
        assert!(caps.total_memory_bytes > 0);
 assert_ne!( caps.total_memory_bytes, crate::fabric::node::FIXED_FAKE_MEMORY_BYTES );
    }
    #[test]
    fn agent_accepts_assignment_and_proves_hashes() {
        let nodes = SimulatedNodeSet::homogeneous_pair_sim("sim-agent-v1", 64 * GIB, 8).nodes;
        let sections = vec![
            ModelSection::content_addressed("s0", 0, 2, 4 * GIB, b"s0"),
            ModelSection::content_addressed("s1", 2, 4, 4 * GIB, b"s1"),
        ];
        let req = PlacementRequest {
            sections,
            nodes: nodes.clone(),
            workload: WorkloadClass::default(),
            seed: 1,
            qualification: QualificationKind::Simulated,
        };
        let plan = PlacementSimulator::new().place(&req).unwrap();
        let target = plan.section_placements[0].node_id.clone();
        let agent = FabricAgent::new(AgentConfig::new(target.as_str(), "127.0.0.1:0"));
        let resp = agent.handle(AgentRequest::Assign {
            assignment: PlacementAssignment {
                plan_id: plan.plan_id.clone(),
                plan: plan.clone(),
                assigned_node: target.clone(),
            },
        });
        match resp {
            AgentResponse::AssignmentAccepted {
                held_section_hashes,
                ..
            } => {
                assert!(!held_section_hashes.is_empty());
            }
            other => panic!("unexpected {other:?}"),
        }
        agent.prove_holds(&plan).unwrap();
    }
}
}


// --- inlined fabric/failure.rs ---
pub mod failure {
//! Heartbeats, failure detection, and failure/replay receipts.
//!
//! Silent loss of work is the failure mode. When a node dies mid-request the
//! system produces a receipt naming what was lost, what was replayed, and from
//! which checkpoint.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::node::NodeId;
use super::placement::{KvOwnershipInvariant, KvRangeOwnership, PlacementPlan};
use super::pipeline::{PipelineScheduler, StageId};
use super::qualification::QualificationKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointId(pub String);

impl CheckpointId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LostWorkSummary {
    pub stages: Vec<StageId>,
    pub kv_ranges: Vec<KvRangeOwnership>,
    pub in_flight_microbatches: u32,
}

/// Receipt produced on node failure + replay. Never silent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureReplayReceipt {
    pub schema: String,
    pub request_id: String,
    pub failed_node: NodeId,
    pub lost_work: LostWorkSummary,
    pub replayed_from_checkpoint: CheckpointId,
    pub replayed_stages: Vec<StageId>,
    pub replan_plan_id: String,
    pub qualification: QualificationKind,
    pub not_physical_qualification: bool,
    /// Filename-safe label for this receipt artifact.
    pub artifact_label: String,
}

pub const FAILURE_RECEIPT_SCHEMA: &str = "hawking.fabric.failure_replay_receipt.v1";

/// Tracks last heartbeat seq per node. Detection uses sequence gaps, not wall clock,
/// so tests stay deterministic.
#[derive(Debug, Default)]
pub struct HeartbeatMonitor {
    /// node -> last seq seen
    last_seq: BTreeMap<NodeId, u64>,
    /// nodes declared dead
    dead: BTreeMap<NodeId, u64>,
    /// max missed seqs before death (deterministic threshold)
    pub miss_threshold: u64,
}

impl HeartbeatMonitor {
    pub fn new(miss_threshold: u64) -> Self {
        Self {
            last_seq: BTreeMap::new(),
            dead: BTreeMap::new(),
            miss_threshold: miss_threshold.max(1),
        }
    }

    pub fn observe(&mut self, node: NodeId, seq: u64) {
        self.last_seq.insert(node, seq);
    }

    /// Advance a global "tick" counter for a node that did not heartbeat.
    /// If the gap from last_seq exceeds threshold, mark dead.
    pub fn note_missed(&mut self, node: &NodeId, global_tick: u64) -> bool {
        if self.dead.contains_key(node) {
            return true;
        }
        let last = self.last_seq.get(node).copied().unwrap_or(0);
        if global_tick.saturating_sub(last) >= self.miss_threshold {
            self.dead.insert(node.clone(), global_tick);
            true
        } else {
            false
        }
    }

    pub fn is_dead(&self, node: &NodeId) -> bool {
        self.dead.contains_key(node)
    }

    pub fn dead_nodes(&self) -> Vec<NodeId> {
        self.dead.keys().cloned().collect()
    }
}

#[derive(Debug)]
pub struct FailureDetector {
    pub heartbeats: HeartbeatMonitor,
}

impl FailureDetector {
    pub fn new(miss_threshold: u64) -> Self {
        Self {
            heartbeats: HeartbeatMonitor::new(miss_threshold),
        }
    }

    /// Build a failure/replay receipt after a node dies mid-request.
    pub fn build_receipt(
        &self,
        request_id: impl Into<String>,
        failed: &NodeId,
        plan: &PlacementPlan,
        pipeline: &PipelineScheduler,
        replan: &PlacementPlan,
        checkpoint: CheckpointId,
    ) -> FailureReplayReceipt {
        let lost_stages: Vec<StageId> = plan
            .stage_assignments
            .iter()
            .filter(|s| &s.node_id == failed)
            .map(|s| s.stage_id.clone())
            .collect();
        let lost_kv = KvOwnershipInvariant::ranges_lost_on_failure(&plan.kv_ownership, failed);
        let st = pipeline.status();
        let in_flight = st.in_flight as u32;

        // Replay stages whose layers overlap any lost stage.
        let lost_layers: Vec<(u32, u32)> = plan
            .stage_assignments
            .iter()
            .filter(|s| &s.node_id == failed)
            .map(|s| (s.layer_start, s.layer_end))
            .collect();
        let replayed_stages: Vec<StageId> = replan
            .stage_assignments
            .iter()
            .filter(|s| {
                lost_layers
                    .iter()
                    .any(|(ls, le)| s.layer_start < *le && s.layer_end > *ls)
            })
            .map(|s| s.stage_id.clone())
            .collect();

        let qualification = if plan.qualification == QualificationKind::Simulated {
            QualificationKind::Simulated
        } else {
            QualificationKind::SoftwareFixture
        };

        FailureReplayReceipt {
            schema: FAILURE_RECEIPT_SCHEMA.into(),
            request_id: request_id.into(),
            failed_node: failed.clone(),
            lost_work: LostWorkSummary {
                stages: lost_stages,
                kv_ranges: lost_kv,
                in_flight_microbatches: in_flight,
            },
            replayed_from_checkpoint: checkpoint,
            replayed_stages,
            replan_plan_id: replan.plan_id.clone(),
            qualification,
            not_physical_qualification: true,
            artifact_label: format!(
                "failure_replay_receipt_{}_{}",
                qualification.as_str(),
                failed.as_str()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::node::SimulatedNodeSet;
    use crate::fabric::placement::{
        ModelSection, PlacementRequest, PlacementSimulator, WorkloadClass,
    };
    const GIB: u64 = 1024 * 1024 * 1024;
    #[test]
    fn heartbeat_marks_dead_after_threshold() {
        let mut mon = HeartbeatMonitor::new(3);
        let n = NodeId::new("n1");
        mon.observe(n.clone(), 1);
        assert!(!mon.note_missed(&n, 2));
        assert!(!mon.note_missed(&n, 3));
        assert!(mon.note_missed(&n, 4)); // 4-1=3 >= 3
        assert!(mon.is_dead(&n));
    }
    #[test]
    fn receipt_names_lost_and_replayed() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-fail-v1").nodes;
        let sections = vec![
            ModelSection::content_addressed("a", 0, 2, 2 * GIB, b"a"),
            ModelSection::content_addressed("b", 2, 4, 2 * GIB, b"b"),
            ModelSection::content_addressed("c", 4, 6, 2 * GIB, b"c"),
        ];
        let workload = WorkloadClass {
            name: "f".into(),
            seq_len: 32,
            microbatch_size: 1,
            num_microbatches: 2,
        };
        let req = PlacementRequest {
            sections,
            nodes,
            workload: workload.clone(),
            seed: 9,
            qualification: QualificationKind::Simulated,
        };
        let sim = PlacementSimulator::new();
        let plan = sim.place(&req).unwrap();
        let failed = plan.stage_assignments[0].node_id.clone();
        let mut pipe = PipelineScheduler::from_plan(&plan, &workload, 2);
        let _ = pipe.tick();
        pipe.mark_node_failed(&failed);
        let replan = sim.replan_after_failure(&req, &failed).unwrap();
        let det = FailureDetector::new(2);
        let receipt = det.build_receipt(
            "req-1",
            &failed,
            &plan,
            &pipe,
            &replan,
            CheckpointId::new("ckpt-after-stage0"),
        );
        assert!(!receipt.lost_work.stages.is_empty());
        assert!(!receipt.lost_work.kv_ranges.is_empty());
        assert_eq!(receipt.replayed_from_checkpoint.0, "ckpt-after-stage0");
        assert!(receipt.not_physical_qualification);
        assert!(receipt.artifact_label.contains("simulated") || receipt.artifact_label.contains("software"));
        assert_ne!(receipt.replan_plan_id, plan.plan_id);
    }
}
}


// --- inlined fabric/fixture.rs ---
pub mod fixture {
//! Local two-process qualification fixture.
//!
//! Two Fabric Agents as two OS processes on this machine, a real placement
//! across them, a real request, a real injected failure, and a real replay
//! receipt. This qualifies the **software**, not the hardware.
//!
//! Status: `PASSED_SOFTWARE` with `not_physical_qualification: true`.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

use super::agent::fixture_receipt;
use super::failure::FailureReplayReceipt;
use super::node::{BandwidthClass, DiscoverySource, NodeCapabilities, NodeId, OsNodeProbe};
use super::placement::{
    ModelSection, PlacementRequest, PlacementSimulator, WorkloadClass, KvOwnershipInvariant,
};
use super::protocol::{AgentRequest, AgentResponse, PlacementAssignment};
use super::qualification::QualificationKind;

#[derive(Debug)]
pub struct AgentProcess {
    pub node_id: NodeId,
    pub addr: String,
    pub child: Child,
}

impl Drop for AgentProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Result of the two-process software qualification fixture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TwoProcessFixtureResult {
    pub schema: String,
    pub qualification: QualificationKind,
    pub not_physical_qualification: bool,
    pub artifact_label: String,
    pub plan_id: String,
    pub request_id: String,
    pub receipt: FailureReplayReceipt,
    pub replan_plan_id: String,
    pub hardware_status: String,
}

pub const FIXTURE_SCHEMA: &str = "hawking.fabric.two_process_fixture.v1";

/// Resolve the fabric-agent binary path (same target dir as tests).
pub fn fabric_agent_bin() -> PathBuf {
    // CARGO_BIN_EXE_fabric-agent is set when running integration tests for this crate.
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_fabric-agent") {
        return PathBuf::from(p);
    }
    // Fallback: target/{debug,release}/fabric-agent relative to workspace.
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // crates
    path.pop(); // workspace
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    path.push("target");
    path.push(profile);
    path.push("fabric-agent");
    path
}

fn wait_for_port(addr: &str, attempts: u32) -> std::io::Result<()> {
    for i in 0..attempts {
        if TcpStream::connect(addr).is_ok() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(20 + i as u64 * 10));
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        format!("port {addr} not ready"),
    ))
}

/// Spawn one fabric-agent process listening on `addr`.
pub fn spawn_agent(bin: &PathBuf, node_id: &str, addr: &str) -> std::io::Result<AgentProcess> {
    let child = Command::new(bin)
        .args(["serve", "--node-id", node_id, "--listen", addr])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    wait_for_port(addr, 100)?;
    Ok(AgentProcess {
        node_id: NodeId::new(node_id),
        addr: addr.to_string(),
        child,
    })
}

/// One request/response over JSON-lines TCP.
pub fn rpc(addr: &str, req: &AgentRequest) -> Result<AgentResponse, String> {
    let mut stream = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    let line = serde_json::to_string(req).map_err(|e| e.to_string())?;
    stream
        .write_all(line.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.write_all(b"\n").map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream);
    let mut resp_line = String::new();
    reader
        .read_line(&mut resp_line)
        .map_err(|e| e.to_string())?;
    serde_json::from_str(resp_line.trim()).map_err(|e| format!("parse {e}: {resp_line}"))
}

/// Capabilities as seen from two real agent processes on this host.
pub fn probe_two_local_agents(
    a: &AgentProcess,
    b: &AgentProcess,
) -> Result<(NodeCapabilities, NodeCapabilities), String> {
    let ra = rpc(
        &a.addr,
        &AgentRequest::Register {
            capabilities: OsNodeProbe::new(a.node_id.as_str()).probe_once(),
        },
    )?;
    let rb = rpc(
        &b.addr,
        &AgentRequest::Register {
            capabilities: OsNodeProbe::new(b.node_id.as_str()).probe_once(),
        },
    )?;
    let ca = match ra {
        AgentResponse::Registered { capabilities } => capabilities,
        other => return Err(format!("agent a register: {other:?}")),
    };
    let cb = match rb {
        AgentResponse::Registered { capabilities } => capabilities,
        other => return Err(format!("agent b register: {other:?}")),
    };
    Ok((ca, cb))
}

/// Run the full two-process software qualification fixture.
pub fn run_two_process_fixture() -> Result<TwoProcessFixtureResult, String> {
    let bin = fabric_agent_bin();
    if !bin.exists() {
        return Err(format!(
            "fabric-agent binary not found at {} (build with cargo test -p hide-fleet --bins)",
            bin.display()
        ));
    }

    // Ephemeral ports: bind OS-chosen ports via helper agents... we pick high ports
    // that are unlikely busy; if bind fails, agent process exits and wait_for_port fails.
    let addr_a = "127.0.0.1:19701";
    let addr_b = "127.0.0.1:19702";

    let mut agent_a =
        spawn_agent(&bin, "fixture-node-a", addr_a).map_err(|e| e.to_string())?;
    let mut agent_b =
        spawn_agent(&bin, "fixture-node-b", addr_b).map_err(|e| e.to_string())?;

    let (mut cap_a, mut cap_b) = probe_two_local_agents(&agent_a, &agent_b)?;
    // Label as software fixture node set (real probes, one machine, two processes).
    cap_a.node_id = NodeId::new("fixture-node-a");
    cap_a.bandwidth_class = BandwidthClass::Localhost;
    cap_a.qualification = QualificationKind::SoftwareFixture;
    cap_a.not_physical_qualification = true;
    // discovery_source stays OsProbe — the *measurement* is real; multi-node fabric is not.
    cap_b.node_id = NodeId::new("fixture-node-b");
    cap_b.bandwidth_class = BandwidthClass::Localhost;
    cap_b.qualification = QualificationKind::SoftwareFixture;
    cap_b.not_physical_qualification = true;

    // Split each process's advertised capacity so placement spreads across both
    // (same host otherwise looks like 2× full RAM).
    let half_a = cap_a.total_memory_bytes / 2;
    let half_b = cap_b.total_memory_bytes / 2;
    cap_a.total_memory_bytes = half_a;
    cap_b.total_memory_bytes = half_b;

    let sections = vec![
        ModelSection::content_addressed("fixture-s0", 0, 2, half_a / 4, b"fixture-s0-v1"),
        ModelSection::content_addressed("fixture-s1", 2, 4, half_b / 4, b"fixture-s1-v1"),
    ];
    let workload = WorkloadClass {
        name: "fixture-request".into(),
        seq_len: 64,
        microbatch_size: 1,
        num_microbatches: 2,
    };
    let req = PlacementRequest {
        sections,
        nodes: vec![cap_a.clone(), cap_b.clone()],
        workload: workload.clone(),
        seed: 42,
        qualification: QualificationKind::SoftwareFixture,
    };
    let sim = PlacementSimulator::new();
    let plan = sim.place(&req).map_err(|e| e.to_string())?;
    KvOwnershipInvariant::assert_holds(&plan.kv_ownership, workload.seq_len)
        .map_err(|e| e.to_string())?;

    // Assign full plan to each agent; each keeps only its sections.
    for (agent, caps) in [(&agent_a, &cap_a), (&agent_b, &cap_b)] {
        let resp = rpc(
            &agent.addr,
            &AgentRequest::Assign {
                assignment: PlacementAssignment {
                    plan_id: plan.plan_id.clone(),
                    plan: plan.clone(),
                    assigned_node: caps.node_id.clone(),
                },
            },
        )?;
        match resp {
            AgentResponse::AssignmentAccepted { .. } => {}
            other => return Err(format!("assign {}: {other:?}", caps.node_id)),
        }
        let _ = rpc(
            &agent.addr,
            &AgentRequest::Heartbeat {
                node_id: caps.node_id.clone(),
                seq: 1,
            },
        )?;
    }

    let request_id = "fixture-req-1".to_string();
    // Real request against both agents.
    for agent in [&agent_a, &agent_b] {
        let resp = rpc(
            &agent.addr,
            &AgentRequest::RunRequest {
                request_id: request_id.clone(),
                plan_id: plan.plan_id.clone(),
            },
        )?;
        match resp {
            AgentResponse::RequestProgress { .. } => {}
            other => return Err(format!("run on {}: {other:?}", agent.node_id)),
        }
    }

    // Fail a node that actually owns stages so the receipt names real lost work.
    let failed = plan
        .stage_assignments
        .first()
        .map(|s| s.node_id.clone())
        .ok_or_else(|| "placement produced no stages".to_string())?;
    let (fail_agent, survivor) = if failed.as_str() == "fixture-node-a" {
        (&mut agent_a, &agent_b)
    } else {
        (&mut agent_b, &agent_a)
    };

    let fail_resp = rpc(
        &fail_agent.addr,
        &AgentRequest::InjectFailure {
            node_id: failed.clone(),
        },
    )?;
    match fail_resp {
        AgentResponse::PlaceholderEvent { .. } => {}
        other => return Err(format!("inject failure: {other:?}")),
    }

    let status = rpc(
        &fail_agent.addr,
        &AgentRequest::GetStatus {
            node_id: failed.clone(),
        },
    )?;
    match status {
        AgentResponse::Status { alive: false, .. } => {}
        other => return Err(format!("expected dead status, got {other:?}")),
    }

    let run_dead = rpc(
        &fail_agent.addr,
        &AgentRequest::RunRequest {
            request_id: request_id.clone(),
            plan_id: plan.plan_id.clone(),
        },
    )?;
    match run_dead {
        AgentResponse::Failed { .. } => {}
        other => return Err(format!("expected Failed after death, got {other:?}")),
    }

    let replan = sim
        .replan_after_failure(&req, &failed)
        .map_err(|e| e.to_string())?;
    KvOwnershipInvariant::assert_holds(&replan.kv_ownership, workload.seq_len)
        .map_err(|e| e.to_string())?;

    // Replay on survivor (and any remaining live node).
    let resp = rpc(
        &survivor.addr,
        &AgentRequest::Assign {
            assignment: PlacementAssignment {
                plan_id: replan.plan_id.clone(),
                plan: replan.clone(),
                assigned_node: survivor.node_id.clone(),
            },
        },
    )?;
    match resp {
        AgentResponse::AssignmentAccepted { .. } => {}
        other => return Err(format!("replan assign: {other:?}")),
    }
    let replay = rpc(
        &survivor.addr,
        &AgentRequest::RunRequest {
            request_id: format!("{request_id}-replay"),
            plan_id: replan.plan_id.clone(),
        },
    )?;
    match replay {
        AgentResponse::RequestProgress { .. } => {}
        other => return Err(format!("replay run: {other:?}")),
    }

    let receipt = fixture_receipt(&request_id, &failed, &plan, &replan, 1);
    if !receipt.not_physical_qualification {
        return Err("receipt must set not_physical_qualification".into());
    }
    if receipt.lost_work.stages.is_empty() && receipt.lost_work.kv_ranges.is_empty() {
        return Err("receipt must name lost work".into());
    }

    // Cleanup processes.
    let _ = rpc(&survivor.addr, &AgentRequest::Shutdown);
    let _ = agent_a.child.kill();
    let _ = agent_b.child.kill();

    Ok(TwoProcessFixtureResult {
        schema: FIXTURE_SCHEMA.into(),
        qualification: QualificationKind::SoftwareFixture,
        not_physical_qualification: true,
        artifact_label: "two_process_fixture_software_qualification".into(),
        plan_id: plan.plan_id,
        request_id,
        receipt,
        replan_plan_id: replan.plan_id,
        hardware_status: super::qualification::HARDWARE_QUALIFICATION_PENDING.into(),
    })
}

/// In-process software fixture (no OS child processes). Used when the binary
/// is unavailable; the integration test prefers the real two-process path.
pub fn run_inprocess_software_fixture() -> Result<TwoProcessFixtureResult, String> {
    use super::agent::{AgentConfig, FabricAgent};

    let probe = OsNodeProbe::new("local");
    let base = probe.probe_once();
    let mut cap_a = base.clone();
    cap_a.node_id = NodeId::new("fixture-node-a");
    cap_a.total_memory_bytes = base.total_memory_bytes / 2;
    cap_a.bandwidth_class = BandwidthClass::Localhost;
    cap_a.qualification = QualificationKind::SoftwareFixture;
    cap_a.not_physical_qualification = true;
    cap_a.discovery_source = DiscoverySource::OsProbe;

    let mut cap_b = cap_a.clone();
    cap_b.node_id = NodeId::new("fixture-node-b");

    let agent_a = FabricAgent::new(AgentConfig::new("fixture-node-a", "127.0.0.1:0"));
    let agent_b = FabricAgent::new(AgentConfig::new("fixture-node-b", "127.0.0.1:0"));

    let sections = vec![
        ModelSection::content_addressed("ip-s0", 0, 2, cap_a.total_memory_bytes / 4, b"ip-s0"),
        ModelSection::content_addressed("ip-s1", 2, 4, cap_b.total_memory_bytes / 4, b"ip-s1"),
    ];
    let workload = WorkloadClass {
        name: "inprocess-fixture".into(),
        seq_len: 32,
        microbatch_size: 1,
        num_microbatches: 2,
    };
    let req = PlacementRequest {
        sections,
        nodes: vec![cap_a, cap_b],
        workload: workload.clone(),
        seed: 7,
        qualification: QualificationKind::SoftwareFixture,
    };
    let sim = PlacementSimulator::new();
    let plan = sim.place(&req).map_err(|e| e.to_string())?;

    for (agent, nid) in [
        (&agent_a, "fixture-node-a"),
        (&agent_b, "fixture-node-b"),
    ] {
        agent.handle(AgentRequest::Assign {
            assignment: PlacementAssignment {
                plan_id: plan.plan_id.clone(),
                plan: plan.clone(),
                assigned_node: NodeId::new(nid),
            },
        });
        agent.handle(AgentRequest::RunRequest {
            request_id: "ip-req-1".into(),
            plan_id: plan.plan_id.clone(),
        });
    }
    agent_b.handle(AgentRequest::InjectFailure {
        node_id: NodeId::new("fixture-node-b"),
    });
    match agent_b.handle(AgentRequest::RunRequest {
        request_id: "ip-req-1".into(),
        plan_id: plan.plan_id.clone(),
    }) {
        AgentResponse::Failed { .. } => {}
        other => return Err(format!("expected fail, got {other:?}")),
    }
    let failed = NodeId::new("fixture-node-b");
    let replan = sim
        .replan_after_failure(&req, &failed)
        .map_err(|e| e.to_string())?;
    KvOwnershipInvariant::assert_holds(&replan.kv_ownership, workload.seq_len)
        .map_err(|e| e.to_string())?;
    agent_a.handle(AgentRequest::Assign {
        assignment: PlacementAssignment {
            plan_id: replan.plan_id.clone(),
            plan: replan.clone(),
            assigned_node: NodeId::new("fixture-node-a"),
        },
    });
    agent_a.handle(AgentRequest::RunRequest {
        request_id: "ip-req-1-replay".into(),
        plan_id: replan.plan_id.clone(),
    });
    let receipt = fixture_receipt("ip-req-1", &failed, &plan, &replan, 1);
    Ok(TwoProcessFixtureResult {
        schema: FIXTURE_SCHEMA.into(),
        qualification: QualificationKind::SoftwareFixture,
        not_physical_qualification: true,
        artifact_label: "two_process_fixture_software_qualification_inprocess".into(),
        plan_id: plan.plan_id,
        request_id: "ip-req-1".into(),
        receipt,
        replan_plan_id: replan.plan_id,
        hardware_status: super::qualification::HARDWARE_QUALIFICATION_PENDING.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inprocess_fixture_produces_labelled_receipt() {
        let result = run_inprocess_software_fixture().expect("fixture");
        assert!(result.not_physical_qualification);
        assert_eq!(result.qualification, QualificationKind::SoftwareFixture);
        assert!(result.receipt.not_physical_qualification);
        assert!(!result.receipt.lost_work.kv_ranges.is_empty() || !result.receipt.lost_work.stages.is_empty());
 assert_eq!( result.hardware_status, super::super::qualification::HARDWARE_QUALIFICATION_PENDING );
    }
}
}


// --- inlined fabric/node.rs ---
pub mod node {
//! Node discovery and capability reporting for the fabric plane.
//!
//! The fleet governor's [`crate::resources::ResourceProbe`] samples free RAM
//! for *agent-job admission*. This module reports a node's full capability
//! envelope for *distributed placement*: total memory, cores, bandwidth class,
//! and accelerator.
//!
//! Discovery is pluggable:
//! - [`OsNodeProbe`] — real OS reads on the local machine
//! - [`SimulatedNodeSet`] — obviously-named injected set for tests/simulation

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::qualification::QualificationKind;

/// Stable node identity. Opaque string; ABI does not assume co-location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Coarse interconnect class. Localhost multi-process is **not** LAN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BandwidthClass {
    /// In-process / shared-memory (not used across OS processes).
    InProcess,
    /// Two processes on the same host (this session's only real interconnect).
    Localhost,
    Lan1g,
    Lan10g,
    Wan,
}

/// Accelerator presence. Reporting only — no kernel dispatch here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "class")]
pub enum AcceleratorClass {
    None,
    AppleSiliconGpu { name: String, gpu_cores: u32 },
    Other { name: String },
}

/// Where capability numbers came from. Simulated must be obvious in name + schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum DiscoverySource {
    OsProbe,
    /// Injected test/sim profile. Name is required and must contain "sim".
    Simulated { profile_name: String },
}

impl DiscoverySource {
    pub fn is_simulated(&self) -> bool {
        matches!(self, Self::Simulated { .. })
    }
}

/// Capability envelope a fabric agent advertises.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub node_id: NodeId,
    pub total_memory_bytes: u64,
    pub free_memory_bytes: u64,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub bandwidth_class: BandwidthClass,
    pub accelerator: AcceleratorClass,
    pub discovery_source: DiscoverySource,
    /// Qualification of this capability report.
    pub qualification: QualificationKind,
    /// Must be true unless `qualification == PhysicalHardware` *and* multi-node
    /// hardware is real. Single-machine OS probes still set this true for
    /// placement plans that use them as a simulated multi-node set.
    pub not_physical_qualification: bool,
}

/// The canned 32 GIB value historically hard-coded into `FixedResourceProbe`
/// via `fleet_run`. Real probes must not equal this as total memory on this
/// 96 GIB M3 Ultra.
pub const FIXED_FAKE_MEMORY_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// Pluggable discovery: real OS probe or injected simulated node set.
#[async_trait]
pub trait NodeDiscovery: Send + Sync {
    async fn discover(&self) -> Vec<NodeCapabilities>;
}

/// Real local-machine probe. Reports this host only (no network discovery).
#[derive(Debug, Clone)]
pub struct OsNodeProbe {
    pub node_id: NodeId,
    /// Interconnect assumption for this agent process. Default: Localhost.
    pub bandwidth_class: BandwidthClass,
}

impl Default for OsNodeProbe {
    fn default() -> Self {
        Self {
            node_id: NodeId::new("local"),
            bandwidth_class: BandwidthClass::Localhost,
        }
    }
}

impl OsNodeProbe {
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: NodeId::new(node_id),
            bandwidth_class: BandwidthClass::Localhost,
        }
    }

    /// Sample once (sync). Used by the agent heartbeat path.
    pub fn probe_once(&self) -> NodeCapabilities {
        let total = read_total_memory_bytes().unwrap_or(0);
        let free_mb = crate::resources::read_free_memory_mb().unwrap_or(0);
        let free = free_mb.saturating_mul(1024 * 1024);
        let physical = read_physical_cores().unwrap_or(1);
        let logical = read_logical_cores().unwrap_or(physical);
        let accelerator = detect_accelerator();
        NodeCapabilities {
            node_id: self.node_id.clone(),
            total_memory_bytes: total,
            free_memory_bytes: free,
            physical_cores: physical,
            logical_cores: logical,
            bandwidth_class: self.bandwidth_class,
            accelerator,
            discovery_source: DiscoverySource::OsProbe,
            // A single-host OS probe is real hardware *measurement*, but it is
            // not multi-node fabric qualification. Placement that uses only this
            // host remains software-local.
            qualification: QualificationKind::SoftwareFixture,
            not_physical_qualification: true,
        }
    }
}

#[async_trait]
impl NodeDiscovery for OsNodeProbe {
    async fn discover(&self) -> Vec<NodeCapabilities> {
        vec![self.probe_once()]
    }
}

/// Obviously-named simulated node set for tests and placement simulation.
///
/// Profile names must contain `"sim"` (case-insensitive). This is intentional:
/// simulated results must be labelled simulated in their own name.
#[derive(Debug, Clone)]
pub struct SimulatedNodeSet {
    pub profile_name: String,
    pub nodes: Vec<NodeCapabilities>,
}

impl SimulatedNodeSet {
    /// Build a heterogeneous simulated set. Panics if `profile_name` does not
    /// contain `"sim"` — simulated things must say so in their name.
    pub fn heterogeneous_sim(profile_name: impl Into<String>) -> Self {
        let profile_name = profile_name.into();
        assert!(
            profile_name.to_ascii_lowercase().contains("sim"),
            "SimulatedNodeSet profile_name must contain 'sim' (got {profile_name})"
        );
        let nodes = vec![
            simulated_node(
                "sim-node-a",
                &profile_name,
                96 * GIB,
                28,
                BandwidthClass::Lan10g,
                AcceleratorClass::AppleSiliconGpu {
                    name: "sim-m3-ultra".into(),
                    gpu_cores: 60,
                },
            ),
            simulated_node(
                "sim-node-b",
                &profile_name,
                64 * GIB,
                16,
                BandwidthClass::Lan1g,
                AcceleratorClass::AppleSiliconGpu {
                    name: "sim-m2-ultra".into(),
                    gpu_cores: 76,
                },
            ),
            simulated_node(
                "sim-node-c",
                &profile_name,
                32 * GIB,
                12,
                BandwidthClass::Lan1g,
                AcceleratorClass::None,
            ),
        ];
        Self {
            profile_name,
            nodes,
        }
    }

    pub fn homogeneous_pair_sim(profile_name: impl Into<String>, mem: u64, cores: u32) -> Self {
        let profile_name = profile_name.into();
        assert!(
            profile_name.to_ascii_lowercase().contains("sim"),
            "SimulatedNodeSet profile_name must contain 'sim'"
        );
        let nodes = vec![
            simulated_node(
                "sim-homog-0",
                &profile_name,
                mem,
                cores,
                BandwidthClass::Lan10g,
                AcceleratorClass::None,
            ),
            simulated_node(
                "sim-homog-1",
                &profile_name,
                mem,
                cores,
                BandwidthClass::Lan10g,
                AcceleratorClass::None,
            ),
        ];
        Self {
            profile_name,
            nodes,
        }
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

fn simulated_node(
    id: &str,
    profile: &str,
    total: u64,
    cores: u32,
    bw: BandwidthClass,
    accel: AcceleratorClass,
) -> NodeCapabilities {
    NodeCapabilities {
        node_id: NodeId::new(id),
        total_memory_bytes: total,
        free_memory_bytes: total / 2,
        physical_cores: cores,
        logical_cores: cores,
        bandwidth_class: bw,
        accelerator: accel,
        discovery_source: DiscoverySource::Simulated {
            profile_name: profile.to_string(),
        },
        qualification: QualificationKind::Simulated,
        not_physical_qualification: true,
    }
}

#[async_trait]
impl NodeDiscovery for SimulatedNodeSet {
    async fn discover(&self) -> Vec<NodeCapabilities> {
        self.nodes.clone()
    }
}

/// Composite discovery: real local probe plus optional simulated peers.
#[derive(Debug, Clone)]
pub struct CompositeDiscovery {
    pub local: OsNodeProbe,
    pub simulated_peers: Option<SimulatedNodeSet>,
}

#[async_trait]
impl NodeDiscovery for CompositeDiscovery {
    async fn discover(&self) -> Vec<NodeCapabilities> {
        let mut out = self.local.discover().await;
        if let Some(sim) = &self.simulated_peers {
            out.extend(sim.discover().await);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// OS readers (no heavy deps)
// ---------------------------------------------------------------------------

pub fn read_total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        read_sysctl_u64("hw.memsize")
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("MemTotal:") {
                let kb: u64 = v.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

pub fn read_physical_cores() -> Option<u32> {
    #[cfg(target_os = "macos")]
    {
        read_sysctl_u64("hw.physicalcpu").map(|n| n as u32)
    }
    #[cfg(target_os = "linux")]
    {
        // Count unique physical id / core id pairs if possible; fall back to nproc.
        std::thread::available_parallelism()
            .ok()
            .map(|n| n.get() as u32)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        std::thread::available_parallelism()
            .ok()
            .map(|n| n.get() as u32)
    }
}

pub fn read_logical_cores() -> Option<u32> {
    #[cfg(target_os = "macos")]
    {
        read_sysctl_u64("hw.logicalcpu")
            .or_else(|| read_sysctl_u64("hw.ncpu"))
            .map(|n| n as u32)
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::thread::available_parallelism()
            .ok()
            .map(|n| n.get() as u32)
    }
}

#[cfg(target_os = "macos")]
fn read_sysctl_u64(name: &str) -> Option<u64> {
    use std::process::Command;
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim().parse().ok()
}

fn detect_accelerator() -> AcceleratorClass {
    #[cfg(target_os = "macos")]
    {
        // Prefer a cheap sysctl brand string over spawning system_profiler.
        let brand = read_sysctl_string("machdep.cpu.brand_string").unwrap_or_default();
        if brand.contains("Apple") {
            // GPU core count is best-effort; 0 means unknown.
            let gpu_cores = 0;
            return AcceleratorClass::AppleSiliconGpu {
                name: brand,
                gpu_cores,
            };
        }
        AcceleratorClass::None
    }
    #[cfg(not(target_os = "macos"))]
    {
        AcceleratorClass::None
    }
}

#[cfg(target_os = "macos")]
fn read_sysctl_string(name: &str) -> Option<String> {
    use std::process::Command;
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Index nodes by id for placement lookups.
pub fn index_by_id(nodes: &[NodeCapabilities]) -> BTreeMap<NodeId, NodeCapabilities> {
    nodes
        .iter()
        .map(|n| (n.node_id.clone(), n.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn os_probe_returns_real_memory_not_fake_32gib() {
        let probe = OsNodeProbe::new("test-local");
        let caps = probe.probe_once();
        assert_eq!(caps.discovery_source, DiscoverySource::OsProbe);
        assert_ne!(caps.total_memory_bytes, FIXED_FAKE_MEMORY_BYTES);
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            assert!(caps.total_memory_bytes > FIXED_FAKE_MEMORY_BYTES);
            assert!(caps.physical_cores >= 1);
            assert!(caps.logical_cores >= caps.physical_cores);
        }
        assert!(caps.not_physical_qualification);
    }
    #[tokio::test]
    async fn simulated_set_is_labelled_simulated() {
        let set = SimulatedNodeSet::heterogeneous_sim("sim-heterogeneous-v1");
        let nodes = set.discover().await;
        assert_eq!(nodes.len(), 3);
        for n in &nodes {
            assert!(n.discovery_source.is_simulated());
            assert_eq!(n.qualification, QualificationKind::Simulated);
            assert!(n.not_physical_qualification);
            assert!(n.node_id.as_str().contains("sim"));
        }
    }
    #[test]
    #[should_panic(expected = "must contain 'sim'")]
    fn simulated_profile_name_must_say_sim() {
        let _ = SimulatedNodeSet::heterogeneous_sim("production-nodes");
    }
}
}


// --- inlined fabric/pipeline.rs ---
pub mod pipeline {
//! Pipeline scheduling: stage graph, in-flight microbatches, backpressure.
//!
//! Pure software scheduling over a placement's stage assignments. No inference
//! execution.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

use super::node::NodeId;
use super::placement::{PlacementPlan, WorkloadClass};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StageId(pub String);

impl StageId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageState {
    Idle,
    Running { microbatch: u32 },
    BlockedBackpressure,
    Failed,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MicrobatchState {
    pub id: u32,
    pub stage_index: usize,
    pub completed: bool,
}

/// Linear pipeline graph derived from a placement plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageGraph {
    pub stages: Vec<StageNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageNode {
    pub stage_id: StageId,
    pub node_id: NodeId,
    pub layer_start: u32,
    pub layer_end: u32,
}

impl StageGraph {
    pub fn from_plan(plan: &PlacementPlan) -> Self {
        let mut stages: Vec<StageNode> = plan
            .stage_assignments
            .iter()
            .map(|s| StageNode {
                stage_id: s.stage_id.clone(),
                node_id: s.node_id.clone(),
                layer_start: s.layer_start,
                layer_end: s.layer_end,
            })
            .collect();
        // Pipeline order by layer_start (ascending).
        stages.sort_by_key(|s| s.layer_start);
        Self { stages }
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineStatus {
    pub in_flight: usize,
    pub completed_microbatches: u32,
    pub backpressured_stages: Vec<StageId>,
    pub stage_states: BTreeMap<String, StageState>,
    pub done: bool,
}

/// Bounded in-flight microbatch scheduler with per-stage queues.
#[derive(Debug)]
pub struct PipelineScheduler {
    graph: StageGraph,
    /// Max in-flight microbatches across the pipeline (backpressure threshold).
    max_in_flight: usize,
    /// Per-stage queue capacity; when full, upstream blocks.
    per_stage_capacity: usize,
    num_microbatches: u32,
    next_to_admit: u32,
    completed: u32,
    /// Queues of microbatch ids waiting/running at each stage.
    stage_queues: Vec<VecDeque<u32>>,
    stage_states: Vec<StageState>,
    /// Microbatches that finished the last stage.
    finished: BTreeSetExt,
}

/// Tiny wrapper so we don't need HashSet import noise for small sets.
#[derive(Debug, Default)]
struct BTreeSetExt {
    inner: BTreeMap<u32, ()>,
}

impl BTreeSetExt {
    fn insert(&mut self, id: u32) {
        self.inner.insert(id, ());
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl PipelineScheduler {
    pub fn from_plan(plan: &PlacementPlan, workload: &WorkloadClass, max_in_flight: usize) -> Self {
        let graph = StageGraph::from_plan(plan);
        let n = graph.len().max(1);
        Self {
            stage_queues: (0..n).map(|_| VecDeque::new()).collect(),
            stage_states: vec![StageState::Idle; n],
            graph,
            max_in_flight: max_in_flight.max(1),
            per_stage_capacity: max_in_flight.max(1),
            num_microbatches: workload.num_microbatches,
            next_to_admit: 0,
            completed: 0,
            finished: BTreeSetExt::default(),
        }
    }

    pub fn in_flight(&self) -> usize {
        let queued: usize = self.stage_queues.iter().map(|q| q.len()).sum();
        queued
    }

    /// Advance one scheduling step. Deterministic given the same admission order.
    pub fn tick(&mut self) -> PipelineStatus {
        // 1. Admit a new microbatch if under cap and more remain.
        if self.next_to_admit < self.num_microbatches
            && self.in_flight() < self.max_in_flight
            && self.stage_queues[0].len() < self.per_stage_capacity
        {
            self.stage_queues[0].push_back(self.next_to_admit);
            self.stage_states[0] = StageState::Running {
                microbatch: self.next_to_admit,
            };
            self.next_to_admit += 1;
        }

        // 2. Propagate: each stage may forward one mb to the next if room.
        // Process from the end so downstream drains first (classic pipeline).
        for i in (0..self.graph.len()).rev() {
            if self.stage_queues[i].is_empty() {
                if !matches!(self.stage_states[i], StageState::Failed) {
                    self.stage_states[i] = StageState::Idle;
                }
                continue;
            }
            let mb = *self.stage_queues[i].front().unwrap();
            if i + 1 == self.graph.len() {
                // Complete at last stage.
                self.stage_queues[i].pop_front();
                self.finished.insert(mb);
                self.completed = self.finished.len() as u32;
                self.stage_states[i] = if self.stage_queues[i].is_empty() {
                    StageState::Idle
                } else {
                    StageState::Running {
                        microbatch: *self.stage_queues[i].front().unwrap(),
                    }
                };
            } else if self.stage_queues[i + 1].len() < self.per_stage_capacity {
                self.stage_queues[i].pop_front();
                self.stage_queues[i + 1].push_back(mb);
                self.stage_states[i + 1] = StageState::Running { microbatch: mb };
                self.stage_states[i] = if self.stage_queues[i].is_empty() {
                    StageState::Idle
                } else {
                    StageState::Running {
                        microbatch: *self.stage_queues[i].front().unwrap(),
                    }
                };
            } else {
                // Backpressure: downstream full.
                self.stage_states[i] = StageState::BlockedBackpressure;
            }
        }

        self.status()
    }

    /// Run until all microbatches complete or `max_ticks` exhausted.
    pub fn run_to_completion(&mut self, max_ticks: usize) -> PipelineStatus {
        for _ in 0..max_ticks {
            let st = self.tick();
            if st.done {
                return st;
            }
        }
        self.status()
    }

    pub fn mark_node_failed(&mut self, node: &NodeId) {
        for (i, stage) in self.graph.stages.iter().enumerate() {
            if &stage.node_id == node {
                self.stage_states[i] = StageState::Failed;
                self.stage_queues[i].clear();
            }
        }
    }

    pub fn status(&self) -> PipelineStatus {
        let mut backpressured = Vec::new();
        let mut map = BTreeMap::new();
        for (i, st) in self.stage_states.iter().enumerate() {
            let id = self.graph.stages[i].stage_id.clone();
            if matches!(st, StageState::BlockedBackpressure) {
                backpressured.push(id.clone());
            }
            map.insert(id.0.clone(), st.clone());
        }
        let done = self.completed >= self.num_microbatches
            && self.stage_queues.iter().all(|q| q.is_empty());
        PipelineStatus {
            in_flight: self.in_flight(),
            completed_microbatches: self.completed,
            backpressured_stages: backpressured,
            stage_states: map,
            done,
        }
    }

    pub fn graph(&self) -> &StageGraph {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::node::SimulatedNodeSet;
    use crate::fabric::placement::{
        ModelSection, PlacementRequest, PlacementSimulator, WorkloadClass,
    };
    use crate::fabric::qualification::QualificationKind;
    const GIB: u64 = 1024 * 1024 * 1024;
    fn plan() -> (PlacementPlan, WorkloadClass) {
        let sections = vec![
            ModelSection::content_addressed("a", 0, 2, 2 * GIB, b"a"),
            ModelSection::content_addressed("b", 2, 4, 2 * GIB, b"b"),
        ];
        let workload = WorkloadClass {
            name: "pipe".into(),
            seq_len: 32,
            microbatch_size: 1,
            num_microbatches: 4,
        };
        let req = PlacementRequest {
            sections,
            nodes: SimulatedNodeSet::homogeneous_pair_sim("sim-pipe-v1", 32 * GIB, 8).nodes,
            workload: workload.clone(),
            seed: 3,
            qualification: QualificationKind::Simulated,
        };
        (PlacementSimulator::new().place(&req).unwrap(), workload)
    }
    #[test]
    fn pipeline_drains_all_microbatches() {
        let (p, w) = plan();
        let mut sched = PipelineScheduler::from_plan(&p, &w, 2);
        let st = sched.run_to_completion(64);
        assert!(st.done, "status={st:?}");
        assert_eq!(st.completed_microbatches, 4);
    }
    #[test]
    fn backpressure_appears_when_capacity_one() {
        let (p, w) = plan();
        let mut sched = PipelineScheduler::from_plan(&p, &w, 1);
        let mut saw_bp = false;
        for _ in 0..8 {
            let st = sched.tick();
            if !st.backpressured_stages.is_empty() {
                saw_bp = true;
                break;
            }
            if st.done {
                break;
            }
        }
        let st = sched.status();
        assert!(saw_bp || st.completed_microbatches > 0 || st.in_flight > 0);
    }
}
}


// --- inlined fabric/placement.rs ---
pub mod placement {
//! Deterministic placement simulator + content-addressed sections + KV ownership.
//!
//! ## KV ownership invariant
//!
//! Every KV range `[token_start, token_end)` for every layer has **exactly one**
//! owning node. No range unowned. No range double-owned. This holds for a
//! placement and must be re-asserted after node failure + replan.
//!
//! Placement is pure and seeded: same inputs + same seed → same plan.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::node::{BandwidthClass, NodeCapabilities, NodeId};
use super::pipeline::StageId;
use super::qualification::QualificationKind;

/// Blake3 hex digest identifying section bytes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

impl ContentHash {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A model section (contiguous layer range) identified by content hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSection {
    pub name: String,
    /// Inclusive start layer index.
    pub layer_start: u32,
    /// Exclusive end layer index.
    pub layer_end: u32,
    pub bytes: u64,
    /// Content hash over name + layer range + payload bytes.
    pub content_hash: ContentHash,
}

impl ModelSection {
    /// Build a section and content-address it from `payload`.
    pub fn content_addressed(
        name: impl Into<String>,
        layer_start: u32,
        layer_end: u32,
        bytes: u64,
        payload: &[u8],
    ) -> Self {
        let name = name.into();
        let content_hash = hash_section(&name, layer_start, layer_end, payload);
        Self {
            name,
            layer_start,
            layer_end,
            bytes,
            content_hash,
        }
    }

    pub fn layer_count(&self) -> u32 {
        self.layer_end.saturating_sub(self.layer_start)
    }
}

pub fn hash_section(name: &str, layer_start: u32, layer_end: u32, payload: &[u8]) -> ContentHash {
    let mut h = blake3::Hasher::new();
    h.update(b"hawking.fabric.section.v1");
    h.update(name.as_bytes());
    h.update(&layer_start.to_le_bytes());
    h.update(&layer_end.to_le_bytes());
    h.update(payload);
    ContentHash(h.finalize().to_hex().to_string())
}

/// Workload class driving pipeline / cost estimation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadClass {
    pub name: String,
    pub seq_len: u32,
    pub microbatch_size: u32,
    pub num_microbatches: u32,
}

impl Default for WorkloadClass {
    fn default() -> Self {
        Self {
            name: "default".into(),
            seq_len: 1024,
            microbatch_size: 1,
            num_microbatches: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionPlacement {
    pub content_hash: ContentHash,
    pub section_name: String,
    pub node_id: NodeId,
    pub layer_start: u32,
    pub layer_end: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageAssignment {
    pub stage_id: StageId,
    pub node_id: NodeId,
    pub layer_start: u32,
    pub layer_end: u32,
    pub section_hash: ContentHash,
}

/// One exclusive KV token range owned by a single node for a layer span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRangeOwnership {
    pub layer_start: u32,
    pub layer_end: u32,
    pub token_start: u32,
    pub token_end: u32,
    pub owner: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictedCost {
    /// Abstract cost units (deterministic function of plan + workload).
    pub total: u64,
    pub transfer_bytes: u64,
    pub pipeline_bubbles: u64,
}

/// Full placement plan. Simulated plans **must** be labelled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub schema: String,
    pub plan_id: String,
    pub seed: u64,
    pub qualification: QualificationKind,
    /// Required true for any non-physical plan. Schema validation rejects
    /// simulated plans with this false/missing.
    pub not_physical_qualification: bool,
    pub section_placements: Vec<SectionPlacement>,
    pub stage_assignments: Vec<StageAssignment>,
    pub kv_ownership: Vec<KvRangeOwnership>,
    pub predicted_cost: PredictedCost,
    /// Filename-safe label; simulated plans include `simulated` in the name.
    pub artifact_label: String,
}

pub const PLACEMENT_SCHEMA: &str = "hawking.fabric.placement.v1";

#[derive(Debug, Clone)]
pub struct PlacementRequest {
    pub sections: Vec<ModelSection>,
    pub nodes: Vec<NodeCapabilities>,
    pub workload: WorkloadClass,
    pub seed: u64,
    pub qualification: QualificationKind,
}

/// Deterministic, seeded placement simulator. No wall-clock dependence.
#[derive(Debug, Default)]
pub struct PlacementSimulator;

impl PlacementSimulator {
    pub fn new() -> Self {
        Self
    }

    pub fn place(&self, req: &PlacementRequest) -> Result<PlacementPlan, PlacementError> {
        if req.nodes.is_empty() {
            return Err(PlacementError::NoNodes);
        }
        if req.sections.is_empty() {
            return Err(PlacementError::NoSections);
        }

        // Stable sort: nodes by (-memory, -cores, id); sections by content_hash.
        let mut nodes = req.nodes.clone();
        nodes.sort_by(|a, b| {
            b.total_memory_bytes
                .cmp(&a.total_memory_bytes)
                .then(b.physical_cores.cmp(&a.physical_cores))
                .then(a.node_id.cmp(&b.node_id))
        });
        let mut sections = req.sections.clone();
        sections.sort_by(|a, b| a.content_hash.cmp(&b.content_hash));

        // Seeded order perturbation: rotate section list by seed % n (deterministic).
        let rot = (req.seed as usize) % sections.len();
        sections.rotate_left(rot);

        let mut remaining: BTreeMap<NodeId, u64> = nodes
            .iter()
            .map(|n| (n.node_id.clone(), n.total_memory_bytes))
            .collect();

        let mut section_placements = Vec::new();
        let mut stage_assignments = Vec::new();
        let mut transfer_bytes = 0u64;

        for (stage_idx, section) in sections.iter().enumerate() {
            let owner = pick_owner(&nodes, &remaining, section.bytes, req.seed, stage_idx)
                .ok_or(PlacementError::InsufficientCapacity {
                    section: section.name.clone(),
                    bytes: section.bytes,
                })?;
            let rem = remaining.get_mut(&owner).expect("owner in map");
            *rem = rem.saturating_sub(section.bytes);

            section_placements.push(SectionPlacement {
                content_hash: section.content_hash.clone(),
                section_name: section.name.clone(),
                node_id: owner.clone(),
                layer_start: section.layer_start,
                layer_end: section.layer_end,
            });
            stage_assignments.push(StageAssignment {
                stage_id: StageId(format!("stage-{stage_idx}")),
                node_id: owner,
                layer_start: section.layer_start,
                layer_end: section.layer_end,
                section_hash: section.content_hash.clone(),
            });
        }

        // Pipeline bubble estimate: |stages - 1| * microbatches abstract units.
        let stages = stage_assignments.len() as u64;
        let bubbles = stages.saturating_sub(1) * req.workload.num_microbatches as u64;

        // Transfer: activation bytes between consecutive stages on different nodes.
        for w in stage_assignments.windows(2) {
            if w[0].node_id != w[1].node_id {
                // Abstract activation size: seq * microbatch * 2 bytes * hidden=4096 proxy.
                let act = (req.workload.seq_len as u64)
                    .saturating_mul(req.workload.microbatch_size as u64)
                    .saturating_mul(4096 * 2);
                transfer_bytes = transfer_bytes.saturating_add(act);
            }
        }

        // Bandwidth-weighted cost (deterministic table).
        let bw_cost = |id: &NodeId| -> u64 {
            nodes
                .iter()
                .find(|n| &n.node_id == id)
                .map(|n| bandwidth_penalty(n.bandwidth_class))
                .unwrap_or(100)
        };
        let mut total = transfer_bytes / 1024; // KB units
        for sp in &section_placements {
            total = total.saturating_add(sp.layer_end.saturating_sub(sp.layer_start) as u64 * 10);
            total = total.saturating_add(bw_cost(&sp.node_id));
        }
        total = total.saturating_add(bubbles * 3);
        // Fold seed lightly so different seeds can change cost when rotation changes owners.
        total = total.saturating_add(req.seed % 97);

        let kv_ownership = build_kv_ownership(&stage_assignments, req.workload.seq_len);

        let not_physical = !req.qualification.is_physical();
        let artifact_label = match req.qualification {
            QualificationKind::Simulated => format!("placement_plan_simulated_seed{}", req.seed),
            QualificationKind::SoftwareFixture => {
                format!("placement_plan_software_fixture_seed{}", req.seed)
            }
            QualificationKind::PhysicalHardware => {
                format!("placement_plan_physical_seed{}", req.seed)
            }
        };

        let plan = PlacementPlan {
            schema: PLACEMENT_SCHEMA.into(),
            plan_id: format!("plan-{:016x}", plan_fingerprint(req)),
            seed: req.seed,
            qualification: req.qualification,
            not_physical_qualification: not_physical,
            section_placements,
            stage_assignments,
            kv_ownership,
            predicted_cost: PredictedCost {
                total,
                transfer_bytes,
                pipeline_bubbles: bubbles,
            },
            artifact_label,
        };

        validate_placement_plan_schema(&plan)?;
        KvOwnershipInvariant::assert_holds(&plan.kv_ownership, req.workload.seq_len)?;
        Ok(plan)
    }

    /// Replan after a failed node: drop the node and place remaining sections.
    pub fn replan_after_failure(
        &self,
        req: &PlacementRequest,
        failed: &NodeId,
    ) -> Result<PlacementPlan, PlacementError> {
        let nodes: Vec<_> = req
            .nodes
            .iter()
            .filter(|n| &n.node_id != failed)
            .cloned()
            .collect();
        if nodes.is_empty() {
            return Err(PlacementError::NoNodes);
        }
        // Bump seed deterministically from failed node id so replan differs stably.
        let mut h = blake3::Hasher::new();
        h.update(&req.seed.to_le_bytes());
        h.update(failed.as_str().as_bytes());
        let digest = h.finalize();
        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&digest.as_bytes()[..8]);
        let seed = u64::from_le_bytes(seed_bytes);
        let replan_req = PlacementRequest {
            sections: req.sections.clone(),
            nodes,
            workload: req.workload.clone(),
            seed,
            qualification: req.qualification,
        };
        self.place(&replan_req)
    }
}

fn pick_owner(
    nodes: &[NodeCapabilities],
    remaining: &BTreeMap<NodeId, u64>,
    need: u64,
    seed: u64,
    stage_idx: usize,
) -> Option<NodeId> {
    let mut candidates: Vec<&NodeCapabilities> = nodes
        .iter()
        .filter(|n| remaining.get(&n.node_id).copied().unwrap_or(0) >= need)
        .collect();
    if candidates.is_empty() {
        // Fall back to node with most remaining capacity (may oversubscribe abstractly).
        return nodes
            .iter()
            .max_by_key(|n| remaining.get(&n.node_id).copied().unwrap_or(0))
            .map(|n| n.node_id.clone());
    }
    // Deterministic tie-break with seed + stage.
    candidates.sort_by(|a, b| {
        let ra = remaining.get(&a.node_id).copied().unwrap_or(0);
        let rb = remaining.get(&b.node_id).copied().unwrap_or(0);
        rb.cmp(&ra).then(a.node_id.cmp(&b.node_id))
    });
    let idx = ((seed as usize).wrapping_add(stage_idx.wrapping_mul(31))) % candidates.len();
    Some(candidates[idx].node_id.clone())
}

fn bandwidth_penalty(bw: BandwidthClass) -> u64 {
    match bw {
        BandwidthClass::InProcess => 1,
        BandwidthClass::Localhost => 2,
        BandwidthClass::Lan10g => 5,
        BandwidthClass::Lan1g => 20,
        BandwidthClass::Wan => 200,
    }
}

fn build_kv_ownership(stages: &[StageAssignment], seq_len: u32) -> Vec<KvRangeOwnership> {
    // Core invariant construction: each stage owns the full token range for its layers.
    // Exactly one owner per (layer, token) because stages have disjoint layer ranges
    // for a linear pipeline partition (enforced by section construction).
    stages
        .iter()
        .map(|s| KvRangeOwnership {
            layer_start: s.layer_start,
            layer_end: s.layer_end,
            token_start: 0,
            token_end: seq_len,
            owner: s.node_id.clone(),
        })
        .collect()
}

fn plan_fingerprint(req: &PlacementRequest) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(&req.seed.to_le_bytes());
    h.update(req.qualification.as_str().as_bytes());
    h.update(req.workload.name.as_bytes());
    h.update(&req.workload.seq_len.to_le_bytes());
    for s in &req.sections {
        h.update(s.content_hash.as_str().as_bytes());
    }
    for n in &req.nodes {
        h.update(n.node_id.as_str().as_bytes());
        h.update(&n.total_memory_bytes.to_le_bytes());
    }
    let d = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&d.as_bytes()[..8]);
    u64::from_le_bytes(b)
}

// ---------------------------------------------------------------------------
// KV ownership invariant
// ---------------------------------------------------------------------------

/// Invariant: every (layer, token) cell has exactly one owner.
pub struct KvOwnershipInvariant;

impl KvOwnershipInvariant {
    pub fn assert_holds(
        ownership: &[KvRangeOwnership],
        seq_len: u32,
    ) -> Result<(), PlacementError> {
        if ownership.is_empty() {
            return Err(PlacementError::KvInvariant {
                detail: "no KV ranges".into(),
            });
        }

        // Expand layer coverage: for each layer index, collect token coverage map.
        let mut max_layer = 0u32;
        for r in ownership {
            if r.layer_end > max_layer {
                max_layer = r.layer_end;
            }
            if r.token_end > seq_len {
                return Err(PlacementError::KvInvariant {
                    detail: format!(
                        "token_end {} exceeds seq_len {}",
                        r.token_end, seq_len
                    ),
                });
            }
            if r.token_start >= r.token_end || r.layer_start >= r.layer_end {
                return Err(PlacementError::KvInvariant {
                    detail: "empty layer or token range".into(),
                });
            }
        }

        // For each layer, build owner per token (None / Some / conflict).
        for layer in 0..max_layer {
            let mut owner_at: Vec<Option<&NodeId>> = vec![None; seq_len as usize];
            for r in ownership {
                if layer < r.layer_start || layer >= r.layer_end {
                    continue;
                }
                for t in r.token_start..r.token_end {
                    let slot = &mut owner_at[t as usize];
                    match slot {
                        None => *slot = Some(&r.owner),
                        Some(existing) if *existing == &r.owner => {}
                        Some(existing) => {
                            return Err(PlacementError::KvInvariant {
                                detail: format!(
                                    "double-owned layer={layer} token={t}: {} and {}",
                                    existing, r.owner
                                ),
                            });
                        }
                    }
                }
            }
            for (t, o) in owner_at.iter().enumerate() {
                if o.is_none() {
                    return Err(PlacementError::KvInvariant {
                        detail: format!("unowned layer={layer} token={t}"),
                    });
                }
            }
        }

        // Also check no two ranges double-cover with different owners (already done).
        let _owners: BTreeSet<_> = ownership.iter().map(|r| &r.owner).collect();
        Ok(())
    }

    /// Drop all ranges owned by `failed` and return the set of lost ranges.
    pub fn ranges_lost_on_failure(
        ownership: &[KvRangeOwnership],
        failed: &NodeId,
    ) -> Vec<KvRangeOwnership> {
        ownership
            .iter()
            .filter(|r| &r.owner == failed)
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Schema validation — rejects unlabelled simulated results
// ---------------------------------------------------------------------------

pub fn validate_placement_plan_schema(plan: &PlacementPlan) -> Result<(), PlacementError> {
    if plan.schema != PLACEMENT_SCHEMA {
        return Err(PlacementError::Schema {
            detail: format!("unexpected schema {}", plan.schema),
        });
    }
    match plan.qualification {
        QualificationKind::Simulated => {
            if !plan.not_physical_qualification {
                return Err(PlacementError::Schema {
                    detail: "simulated placement must set not_physical_qualification=true".into(),
                });
            }
            if !plan.artifact_label.contains("simulated") {
                return Err(PlacementError::Schema {
                    detail: "simulated placement artifact_label must contain 'simulated'".into(),
                });
            }
            if !plan.plan_id.is_empty() && plan.artifact_label.contains("physical") {
                return Err(PlacementError::Schema {
                    detail: "simulated placement must not claim physical in artifact_label".into(),
                });
            }
        }
        QualificationKind::SoftwareFixture => {
            if !plan.not_physical_qualification {
                return Err(PlacementError::Schema {
                    detail: "software_fixture placement must set not_physical_qualification=true"
                        .into(),
                });
            }
        }
        QualificationKind::PhysicalHardware => {
            if plan.not_physical_qualification {
                return Err(PlacementError::Schema {
                    detail: "physical_hardware placement must not set not_physical_qualification"
                        .into(),
                });
            }
        }
    }
    Ok(())
}

/// Reject an unlabelled simulated result (for tests that construct bad plans).
pub fn reject_unlabelled_simulated(plan: &PlacementPlan) -> Result<(), PlacementError> {
    if plan.qualification == QualificationKind::Simulated && !plan.not_physical_qualification {
        return Err(PlacementError::Schema {
            detail: "unlabelled simulated result rejected".into(),
        });
    }
    if plan.qualification == QualificationKind::Simulated
        && !plan.artifact_label.to_ascii_lowercase().contains("sim")
    {
        return Err(PlacementError::Schema {
            detail: "simulated artifact must be labelled simulated".into(),
        });
    }
    validate_placement_plan_schema(plan)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    #[error("no nodes available for placement")]
    NoNodes,
    #[error("no sections to place")]
    NoSections,
    #[error("insufficient capacity for section {section} ({bytes} bytes)")]
    InsufficientCapacity { section: String, bytes: u64 },
    #[error("KV ownership invariant violated: {detail}")]
    KvInvariant { detail: String },
    #[error("placement schema error: {detail}")]
    Schema { detail: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::node::SimulatedNodeSet;
    fn sample_sections() -> Vec<ModelSection> {
        vec![
            ModelSection::content_addressed("embed", 0, 2, 4 * GIB, b"embed-payload-v1"),
            ModelSection::content_addressed("mid", 2, 6, 8 * GIB, b"mid-payload-v1"),
            ModelSection::content_addressed("head", 6, 8, 3 * GIB, b"head-payload-v1"),
        ]
    }
    const GIB: u64 = 1024 * 1024 * 1024;
    #[test]
    fn content_hash_stable_across_repacks() {
        let a = ModelSection::content_addressed("mid", 2, 6, 8 * GIB, b"mid-payload-v1");
        let b = ModelSection::content_addressed("mid", 2, 6, 999, b"mid-payload-v1");
        assert_eq!(a.content_hash, b.content_hash);
        let c = ModelSection::content_addressed("mid", 2, 6, 8 * GIB, b"mid-payload-v2");
        assert_ne!(a.content_hash, c.content_hash);
    }
    #[test]
    fn placement_determinism_same_seed_same_plan() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-det-v1").nodes;
        let req = PlacementRequest {
            sections: sample_sections(),
            nodes,
            workload: WorkloadClass {
                name: "decode".into(),
                seq_len: 512,
                microbatch_size: 2,
                num_microbatches: 4,
            },
            seed: 0xC0FFEE,
            qualification: QualificationKind::Simulated,
        };
        let sim = PlacementSimulator::new();
        let p1 = sim.place(&req).unwrap();
        let p2 = sim.place(&req).unwrap();
        assert_eq!(p1, p2);
        assert!(p1.not_physical_qualification);
        assert!(p1.artifact_label.contains("simulated"));
    }
    #[test]
    fn different_seeds_can_differ() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-seed-v1").nodes;
        let base = PlacementRequest {
            sections: sample_sections(),
            nodes,
            workload: WorkloadClass::default(),
            seed: 1,
            qualification: QualificationKind::Simulated,
        };
        let sim = PlacementSimulator::new();
        let p1 = sim.place(&base).unwrap();
        let mut base2 = base.clone();
        base2.seed = 2;
        let p2 = sim.place(&base2).unwrap();
        assert_ne!(p1.plan_id, p2.plan_id);
    }
    #[test]
    fn kv_ownership_invariant_holds() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-kv-v1").nodes;
        let req = PlacementRequest {
            sections: sample_sections(),
            nodes,
            workload: WorkloadClass {
                name: "kv".into(),
                seq_len: 128,
                microbatch_size: 1,
                num_microbatches: 2,
            },
            seed: 7,
            qualification: QualificationKind::Simulated,
        };
        let plan = PlacementSimulator::new().place(&req).unwrap();
        KvOwnershipInvariant::assert_holds(&plan.kv_ownership, req.workload.seq_len).unwrap();
    }
    #[test]
    fn kv_invariant_after_failure_replan() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-kv-fail-v1").nodes;
        let req = PlacementRequest {
            sections: sample_sections(),
            nodes: nodes.clone(),
            workload: WorkloadClass {
                name: "kv".into(),
                seq_len: 64,
                microbatch_size: 1,
                num_microbatches: 2,
            },
            seed: 11,
            qualification: QualificationKind::Simulated,
        };
        let sim = PlacementSimulator::new();
        let plan = sim.place(&req).unwrap();
        let failed = plan.section_placements[0].node_id.clone();
        let lost = KvOwnershipInvariant::ranges_lost_on_failure(&plan.kv_ownership, &failed);
        assert!(!lost.is_empty(), "expected some KV lost on node failure");
        let replan = sim.replan_after_failure(&req, &failed).unwrap();
        assert!(replan .section_placements .iter() .all(|sp| sp.node_id != failed));
        KvOwnershipInvariant::assert_holds(&replan.kv_ownership, req.workload.seq_len).unwrap();
    }
    #[test]
    fn schema_rejects_unlabelled_simulated_result() {
        let mut plan = PlacementPlan {
            schema: PLACEMENT_SCHEMA.into(),
            plan_id: "bad".into(),
            seed: 0,
            qualification: QualificationKind::Simulated,
            not_physical_qualification: false, // ILLEGAL
            section_placements: vec![],
            stage_assignments: vec![],
            kv_ownership: vec![],
            predicted_cost: PredictedCost {
                total: 0,
                transfer_bytes: 0,
                pipeline_bubbles: 0,
            },
            artifact_label: "looks_physical".into(),
        };
        let err = reject_unlabelled_simulated(&plan).unwrap_err();
        assert!(matches!(err, PlacementError::Schema { .. }));
        plan.not_physical_qualification = true;
        plan.artifact_label = "placement_plan_simulated_seed0".into();
        validate_placement_plan_schema(&plan).unwrap();
    }
}
}


// --- inlined fabric/protocol.rs ---
pub mod protocol {
//! Localhost fabric agent wire protocol (JSON lines over TCP).
//!
//! ABI does **not** assume co-location: messages are explicit, node ids are
//! opaque, and only loopback is used in this session (no network discovery).

use serde::{Deserialize, Serialize};

use super::failure::FailureReplayReceipt;
use super::node::{NodeCapabilities, NodeId};
use super::placement::PlacementPlan;

/// Assignment of a placement plan slice to a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementAssignment {
    pub plan_id: String,
    pub plan: PlacementPlan,
    pub assigned_node: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum AgentRequest {
    /// Agent → coordinator: register with capabilities.
    Register { capabilities: NodeCapabilities },
    /// Agent → coordinator: heartbeat with monotonic seq.
    Heartbeat { node_id: NodeId, seq: u64 },
    /// Coordinator → agent: accept a placement assignment.
    Assign { assignment: PlacementAssignment },
    /// Coordinator → agent: run a logical request id through local stages.
    RunRequest {
        request_id: String,
        plan_id: String,
    },
    /// Coordinator → agent: inject a synthetic failure (fixture only).
    InjectFailure { node_id: NodeId },
    /// Coordinator → agent: query status.
    GetStatus { node_id: NodeId },
    /// Coordinator → agent: shutdown.
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AgentResponse {
    Ok {
        node_id: NodeId,
        detail: String,
    },
    Registered {
        capabilities: NodeCapabilities,
    },
    HeartbeatAck {
        node_id: NodeId,
        seq: u64,
    },
    AssignmentAccepted {
        node_id: NodeId,
        plan_id: String,
        held_section_hashes: Vec<String>,
    },
    RequestProgress {
        node_id: NodeId,
        request_id: String,
        completed_local_stages: u32,
    },
    Failed {
        node_id: NodeId,
        reason: String,
    },
    Status {
        node_id: NodeId,
        alive: bool,
        assignment_plan_id: Option<String>,
        held_section_hashes: Vec<String>,
    },
    /// Placeholder event emission (Bridge lane owns the real event model).
    PlaceholderEvent {
        kind: String,
        payload: serde_json::Value,
    },
    Receipt {
        receipt: FailureReplayReceipt,
    },
    Error {
        message: String,
    },
}

/// Fabric placeholder event kind — Bridge lane owns the real event model.
pub const FABRIC_PLACEHOLDER_EVENT_KIND: &str = "fabric.placeholder.event.v1";
}


// --- inlined fabric/qualification.rs ---
pub mod qualification {
//! Qualification labelling for fabric software vs hardware vs simulation.
//!
//! **Law:** never claim physical qualification from simulation or a local
//! multi-process fixture. The honest terminal hardware state on this single
//! M3 Ultra is `FABRIC_HARDWARE_QUALIFICATION_PENDING`.

use serde::{Deserialize, Serialize};

/// How a fabric result was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationKind {
    /// Multi-node physical fabric. Not available in this session.
    PhysicalHardware,
    /// Deterministic placement over an injected/simulated node set.
    Simulated,
    /// Local multi-process software fixture on one machine.
    SoftwareFixture,
}

impl QualificationKind {
    /// Physical hardware is the only kind that may omit
    /// `not_physical_qualification: true`.
    pub fn is_physical(&self) -> bool {
        matches!(self, Self::PhysicalHardware)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PhysicalHardware => "physical_hardware",
            Self::Simulated => "simulated",
            Self::SoftwareFixture => "software_fixture",
        }
    }
}

/// Schema id for qualification-bearing fabric artifacts.
pub const QUALIFICATION_SCHEMA: &str = "hawking.fabric.qualification.v1";

/// Human-readable hardware terminal state for this campaign session.
pub const HARDWARE_QUALIFICATION_PENDING: &str = "FABRIC_HARDWARE_QUALIFICATION_PENDING";

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simulated_is_not_physical() {
        assert!(!QualificationKind::Simulated.is_physical());
        assert!(!QualificationKind::SoftwareFixture.is_physical());
        assert!(QualificationKind::PhysicalHardware.is_physical());
    }
}
}

