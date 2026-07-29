//! # hide-you — YOU surface swarms, projects, and typed handoffs
//!
//! HIDE has three surfaces (YOU / CHAT / IDE) as lenses over one session. This
//! crate implements the YOU-side product layer that the surface-authority
//! contract requires:
//!
//! * **Swarms** — governed teams on the fleet substrate concept (roles, modes,
//!   per-agent capsules, budgets), not prompt multiplication.
//! * **Projects** — unified containers for conversations, documents, objects,
//!   connectors, plans, tasks, memory, automations, agents, and artifacts.
//! * **Typed handoffs** — YOU→CHAT / CHAT→IDE / IDE→YOU capsules that carry
//!   **claims**, never **capabilities**.
//!
//! ## The properties that matter
//!
//! 1. A capsule carries the CLAIM, never the CAPABILITY. Receiving a YOU→CHAT
//!    capsule must not grant CHAT the connector access YOU held.
//! 2. No agent promotes its own high-risk conclusion; promotion needs
//!    independent verification. Consensus is weak; a reproduced defect outranks
//!    votes.
//! 3. Resource economics are enforced (CPU/RAM, tokens/steps, wall time, stop
//!    condition). A swarm that exceeds its budget halts and records why.
//! 4. Every capsule carries provenance, evidence tier of claims, permissions it
//!    was created under, and what it deliberately excludes.
//!
//! ## Why a new crate (not hide-fleet)
//!
//! YOU-surface domain (roles, modes, projects, typed surface handoffs) is
//! orthogonal to hide-fleet's machine-wide job scheduler and GPU admission;
//! this crate reuses the capability-derivation pattern from
//! `hide_core::automation` and does not pull hide-kernel or Metal.
//!
//! ## What is fake
//!
//! All "inference" goes through [`fixture::FixtureProvider`]. No model loads,
//! no network, no Metal, no Odyssey fence, no adapter-grade changes.

pub use agent::{AgentId, AgentReceipt, AgentSpec, OutputSchema, VerificationContract};
pub use budget::{
    BudgetAxis, BudgetUsage, ResourceBudget, StopCondition, StopReason, SwarmBudget,
};
pub use capsule::{
    Claim, DeliberateExclusion, HandoffCapsule, HandoffKind, OpenedCapsule, PermissionSnapshot,
    ProvenanceEntry, ReceivedHandoff, SurfaceSession,
};
pub use capability::{CapabilitySnapshot, SurfaceCapability, SurfacePermissionSet};
pub use error::{Result, YouError};
pub use evidence::EvidenceTier;
pub use fixture::{FixtureProvider, FixtureReply};
pub use modes::SwarmMode;
pub use project::{Project, ProjectId, ProjectMemberKind, ProjectState};
pub use promotion::{
    Conclusion, ConclusionRisk, PromotionBoard, PromotionDecision, PromotionEvidence, VoteTally,
};
pub use roles::AgentRole;
pub use session_graph::{
    CapsuleView, LensView, OpenedCapsuleView, SurfaceGraph, SurfaceGraphView,
};
pub use surface::{Surface, SurfaceDefaults};
pub use swarm::{Swarm, SwarmId, SwarmStatus};

// --- inlined lenses/agent.rs ---
pub mod agent {
//! Swarm agent specification and fixture receipts.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lenses::budget::{ResourceBudget, StopCondition};
use crate::lenses::capability::SurfaceCapability;
use crate::lenses::evidence::EvidenceTier;
use crate::lenses::roles::AgentRole;

/// Stable agent id (`agt_…`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new() -> Self {
        Self(format!("agt_{}", ulid::Ulid::new().to_string().to_ascii_lowercase()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Declared output shape an agent must produce (schema id + optional JSON Schema).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputSchema {
    pub schema_id: String,
    #[serde(default)]
    pub json_schema: Value,
}

impl OutputSchema {
    pub fn named(schema_id: impl Into<String>) -> Self {
        Self {
            schema_id: schema_id.into(),
            json_schema: Value::Null,
        }
    }
}

/// What independent verification is required for this agent's outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationContract {
    /// Minimum evidence tier before a high-risk claim may leave the agent.
    pub min_high_risk_tier: EvidenceTier,
    /// Whether a distinct Verifier role is required for promotion.
    pub require_independent_verifier: bool,
    /// Whether a reproduced defect/oracle outranks consensus votes.
    pub reproduction_outranks_consensus: bool,
}

impl Default for VerificationContract {
    fn default() -> Self {
        Self {
            min_high_risk_tier: EvidenceTier::IndependentlyVerified,
            require_independent_verifier: true,
            reproduction_outranks_consensus: true,
        }
    }
}

/// Full agent brief: everything a swarm member receives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSpec {
    pub id: AgentId,
    pub goal: String,
    pub role: AgentRole,
    /// Context capsule id or free-form context payload (claims, not capabilities).
    pub context_capsule: Value,
    /// Model/profile label (fixture only; no real routing).
    pub model_profile: String,
    pub tools: Vec<String>,
    pub connectors: Vec<String>,
    /// Capability derived from the swarm's permission set (non-widening).
    pub permissions: SurfaceCapability,
    pub budget: ResourceBudget,
    pub deadline_ms: Option<u64>,
    pub output_schema: OutputSchema,
    pub verification: VerificationContract,
    pub stop: StopCondition,
}

impl AgentSpec {
    pub fn builder(role: AgentRole, goal: impl Into<String>) -> AgentSpecBuilder {
        AgentSpecBuilder {
            role,
            goal: goal.into(),
            context_capsule: Value::Null,
            model_profile: "fixture/general".into(),
            tools: Vec::new(),
            connectors: Vec::new(),
            permissions: SurfaceCapability::default(),
            budget: ResourceBudget::default(),
            deadline_ms: None,
            output_schema: OutputSchema::named("default"),
            verification: VerificationContract::default(),
            stop: StopCondition::Never,
        }
    }
}

/// Fluent builder for [`AgentSpec`].
pub struct AgentSpecBuilder {
    role: AgentRole,
    goal: String,
    context_capsule: Value,
    model_profile: String,
    tools: Vec<String>,
    connectors: Vec<String>,
    permissions: SurfaceCapability,
    budget: ResourceBudget,
    deadline_ms: Option<u64>,
    output_schema: OutputSchema,
    verification: VerificationContract,
    stop: StopCondition,
}

impl AgentSpecBuilder {
    pub fn context(mut self, ctx: Value) -> Self {
        self.context_capsule = ctx;
        self
    }

    pub fn model_profile(mut self, profile: impl Into<String>) -> Self {
        self.model_profile = profile.into();
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tools = tools.into_iter().map(Into::into).collect();
        self
    }

    pub fn connectors(mut self, c: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.connectors = c.into_iter().map(Into::into).collect();
        self
    }

    pub fn permissions(mut self, cap: SurfaceCapability) -> Self {
        self.permissions = cap;
        self
    }

    pub fn budget(mut self, budget: ResourceBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn deadline_ms(mut self, ms: u64) -> Self {
        self.deadline_ms = Some(ms);
        self
    }

    pub fn output_schema(mut self, schema: OutputSchema) -> Self {
        self.output_schema = schema;
        self
    }

    pub fn verification(mut self, v: VerificationContract) -> Self {
        self.verification = v;
        self
    }

    pub fn stop(mut self, stop: StopCondition) -> Self {
        self.stop = stop;
        self
    }

    pub fn build(self) -> AgentSpec {
        AgentSpec {
            id: AgentId::new(),
            goal: self.goal,
            role: self.role,
            context_capsule: self.context_capsule,
            model_profile: self.model_profile,
            tools: self.tools,
            connectors: self.connectors,
            permissions: self.permissions,
            budget: self.budget,
            deadline_ms: self.deadline_ms,
            output_schema: self.output_schema,
            verification: self.verification,
            stop: self.stop,
        }
    }
}

/// Deterministic fixture receipt from one agent step (no real inference).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentReceipt {
    pub agent_id: AgentId,
    pub role: AgentRole,
    pub ok: bool,
    pub summary: String,
    pub tokens_used: u64,
    pub steps_used: u32,
    pub cpu_ms: u64,
    pub ram_mb: u64,
    pub claims: Vec<crate::lenses::capsule::Claim>,
    pub evidence_tier: EvidenceTier,
}
}


// --- inlined lenses/budget.rs ---
pub mod budget {
//! Resource economics for swarms and agents.
//!
//! More agents is not automatically better. Exhaustion of any budget axis is a
//! hard stop; the swarm records which axis fired.

use serde::{Deserialize, Serialize};

/// Which budget axis was exhausted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetAxis {
    CpuMs,
    RamMb,
    Tokens,
    Steps,
    WallMs,
    AgentCount,
}

impl BudgetAxis {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CpuMs => "cpu_ms",
            Self::RamMb => "ram_mb",
            Self::Tokens => "tokens",
            Self::Steps => "steps",
            Self::WallMs => "wall_ms",
            Self::AgentCount => "agent_count",
        }
    }
}

/// Resource bounds. Exhaustion is a hard stop.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub max_cpu_ms: Option<u64>,
    pub max_ram_mb: Option<u64>,
    pub max_tokens: Option<u64>,
    pub max_steps: Option<u32>,
    pub max_wall_ms: Option<u64>,
    /// Soft ceiling on concurrent/total agents for the swarm.
    pub max_agents: Option<u32>,
}

/// Cumulative spend against a budget.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BudgetUsage {
    pub cpu_ms: u64,
    pub ram_mb_peak: u64,
    pub tokens: u64,
    pub steps: u32,
    pub wall_ms: u64,
    pub agents_launched: u32,
}

impl BudgetUsage {
    /// Which axis, if any, is exhausted under `budget`.
    pub fn exhausted_axis(&self, budget: &ResourceBudget) -> Option<BudgetAxis> {
        if budget.max_cpu_ms.is_some_and(|m| self.cpu_ms >= m) {
            return Some(BudgetAxis::CpuMs);
        }
        if budget.max_ram_mb.is_some_and(|m| self.ram_mb_peak >= m) {
            return Some(BudgetAxis::RamMb);
        }
        if budget.max_tokens.is_some_and(|m| self.tokens >= m) {
            return Some(BudgetAxis::Tokens);
        }
        if budget.max_steps.is_some_and(|m| self.steps >= m) {
            return Some(BudgetAxis::Steps);
        }
        if budget.max_wall_ms.is_some_and(|m| self.wall_ms >= m) {
            return Some(BudgetAxis::WallMs);
        }
        if budget.max_agents.is_some_and(|m| self.agents_launched > m) {
            return Some(BudgetAxis::AgentCount);
        }
        None
    }

    pub fn is_exhausted(&self, budget: &ResourceBudget) -> bool {
        self.exhausted_axis(budget).is_some()
    }
}

/// When a swarm or agent must halt. Enforced, not advisory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum StopCondition {
    Never,
    AfterSteps { count: u32 },
    AfterWallMs { ms: u64 },
    ConditionMet { name: String },
    BudgetOnly,
}

/// Why a swarm or agent halted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum StopReason {
    BudgetExhausted { axis: String },
    AfterSteps { count: u32 },
    AfterWallMs { ms: u64 },
    ConditionMet { name: String },
    Cancelled,
    AuthorityDenied { what: String },
    Completed,
}

/// Swarm-level budget + stop condition package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmBudget {
    pub resources: ResourceBudget,
    pub stop: StopCondition,
}

impl Default for SwarmBudget {
    fn default() -> Self {
        Self {
            resources: ResourceBudget {
                max_cpu_ms: Some(60_000),
                max_ram_mb: Some(512),
                max_tokens: Some(50_000),
                max_steps: Some(100),
                max_wall_ms: Some(300_000),
                max_agents: Some(8),
            },
            stop: StopCondition::BudgetOnly,
        }
    }
}

impl SwarmBudget {
    pub fn tight_fixture() -> Self {
        Self {
            resources: ResourceBudget {
                max_cpu_ms: Some(100),
                max_ram_mb: Some(64),
                max_tokens: Some(50),
                max_steps: Some(3),
                max_wall_ms: Some(1_000),
                max_agents: Some(4),
            },
            stop: StopCondition::BudgetOnly,
        }
    }
}
}


// --- inlined lenses/capability.rs ---
pub mod capability {
//! Surface capability derivation — structurally non-widening.
//!
//! Mirrors `hide_core::automation::{PermissionSet, JobCapability}`: a
//! [`SurfaceCapability`] can only be obtained by deriving from a
//! [`SurfacePermissionSet`]. There is no public constructor that invents
//! tools or connectors, and no method that adds them after the fact.
//!
//! This is the enforcement spine for the handoff invariant: a capsule may
//! *describe* permissions held at creation, but cannot mint a capability.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::lenses::error::{Result, YouError};

/// Closed set of tools and connectors a surface (or agent) is allowed to use.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SurfacePermissionSet {
    tools: BTreeSet<String>,
    connectors: BTreeSet<String>,
}

impl SurfacePermissionSet {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn new(
        tools: impl IntoIterator<Item = impl Into<String>>,
        connectors: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            tools: tools.into_iter().map(Into::into).collect(),
            connectors: connectors.into_iter().map(Into::into).collect(),
        }
    }

    pub fn tools(&self) -> &BTreeSet<String> {
        &self.tools
    }

    pub fn connectors(&self) -> &BTreeSet<String> {
        &self.connectors
    }

    pub fn grants_tool(&self, name: &str) -> bool {
        self.tools.contains(name)
    }

    pub fn grants_connector(&self, name: &str) -> bool {
        self.connectors.contains(name)
    }

    /// Full capability granted by this set. Receiver cannot widen it.
    pub fn derive_capability(&self) -> SurfaceCapability {
        SurfaceCapability {
            tools: self.tools.clone(),
            connectors: self.connectors.clone(),
            live: true,
        }
    }

    /// Subset derivation; requesting anything outside the set fails closed.
    pub fn derive_capability_subset(
        &self,
        tools: impl IntoIterator<Item = impl AsRef<str>>,
        connectors: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<SurfaceCapability> {
        let mut out_tools = BTreeSet::new();
        for t in tools {
            let name = t.as_ref();
            if !self.tools.contains(name) {
                return Err(YouError::CapabilityMissing(format!(
                    "cannot derive capability for tool '{name}': not in permission set"
                )));
            }
            out_tools.insert(name.to_string());
        }
        let mut out_connectors = BTreeSet::new();
        for c in connectors {
            let name = c.as_ref();
            if !self.connectors.contains(name) {
                return Err(YouError::CapabilityMissing(format!(
                    "cannot derive capability for connector '{name}': not in permission set"
                )));
            }
            out_connectors.insert(name.to_string());
        }
        Ok(SurfaceCapability {
            tools: out_tools,
            connectors: out_connectors,
            live: true,
        })
    }

    /// Intersection with another set (for session-scoped narrowing only).
    pub fn intersect(&self, other: &SurfacePermissionSet) -> SurfacePermissionSet {
        SurfacePermissionSet {
            tools: self
                .tools
                .intersection(&other.tools)
                .cloned()
                .collect(),
            connectors: self
                .connectors
                .intersection(&other.connectors)
                .cloned()
                .collect(),
        }
    }
}

/// Capability handed to a surface session or agent. **Structurally
/// non-widening**: fields private; only construction path is
/// [`SurfacePermissionSet::derive_capability`] /
/// [`SurfacePermissionSet::derive_capability_subset`].
///
/// `live` is never serialized. A capability forged via `serde` (or any other
/// path that does not go through derive) deserializes with `live = false` and
/// fails closed on every gate. That closes the export / handoff payload attack
/// of smuggling a capability-shaped JSON object.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SurfaceCapability {
    tools: BTreeSet<String>,
    connectors: BTreeSet<String>,
    /// True only when constructed by [`SurfacePermissionSet::derive_capability`]
    /// or [`SurfacePermissionSet::derive_capability_subset`].
    #[serde(skip)]
    live: bool,
}

impl SurfaceCapability {
    pub fn tools(&self) -> &BTreeSet<String> {
        &self.tools
    }

    pub fn connectors(&self) -> &BTreeSet<String> {
        &self.connectors
    }

    /// Whether this handle was minted by derive (not forged via serde/export).
    pub fn is_live(&self) -> bool {
        self.live
    }

    pub fn allows_tool(&self, name: &str) -> bool {
        self.live && self.tools.contains(name)
    }

    pub fn allows_connector(&self, name: &str) -> bool {
        self.live && self.connectors.contains(name)
    }

    pub fn require_tool(&self, name: &str) -> Result<()> {
        if !self.live {
            return Err(YouError::PolicyDenied(
                "surface capability is not live (forged or deserialized; derive only)".into(),
            ));
        }
        if self.tools.contains(name) {
            Ok(())
        } else {
            Err(YouError::PolicyDenied(format!(
                "surface capability does not grant tool '{name}'"
            )))
        }
    }

    pub fn require_connector(&self, name: &str) -> Result<()> {
        if !self.live {
            return Err(YouError::PolicyDenied(
                "surface capability is not live (forged or deserialized; derive only)".into(),
            ));
        }
        if self.connectors.contains(name) {
            Ok(())
        } else {
            Err(YouError::PolicyDenied(format!(
                "surface capability does not grant connector '{name}'"
            )))
        }
    }

    /// True iff every tool/connector in `self` is also in `parent`.
    pub fn is_within(&self, parent: &SurfacePermissionSet) -> bool {
        self.live
            && self.tools.is_subset(parent.tools())
            && self.connectors.is_subset(parent.connectors())
    }

    /// Snapshot for audit (claims about authority, not a grant handle).
    pub fn snapshot(&self) -> CapabilitySnapshot {
        CapabilitySnapshot {
            tools: self.tools.iter().cloned().collect(),
            connectors: self.connectors.iter().cloned().collect(),
        }
    }
}

/// Serializable description of tools/connectors held at a moment in time.
/// This is a CLAIM about authority for provenance — not a live capability.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CapabilitySnapshot {
    pub tools: Vec<String>,
    pub connectors: Vec<String>,
}

impl CapabilitySnapshot {
    pub fn from_set(set: &SurfacePermissionSet) -> Self {
        Self {
            tools: set.tools().iter().cloned().collect(),
            connectors: set.connectors().iter().cloned().collect(),
        }
    }

    pub fn from_capability(cap: &SurfaceCapability) -> Self {
        cap.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn derive_is_live_default_is_not() {
        let empty = SurfaceCapability::default();
        assert!(!empty.is_live());
        assert!(!empty.allows_tool("x"));
        let set = SurfacePermissionSet::new(["t1"], ["c1"]);
        let cap = set.derive_capability();
        assert!(cap.is_live());
        assert!(cap.allows_tool("t1"));
        assert!(cap.allows_connector("c1"));
    }
    #[test]
    fn adversarial_serde_forge_is_not_live() {
        let forged: SurfaceCapability = serde_json::from_value(json!({
            "tools": ["shell.exec"],
            "connectors": ["gmail"],
            "live": true
        }))
        .unwrap();
        assert!(!forged.is_live());
        assert!(!forged.allows_tool("shell.exec"));
        assert!(forged.require_tool("shell.exec").is_err());
        assert!(forged.require_connector("gmail").is_err());
    }
    #[test]
    fn subset_cannot_widen() {
        let set = SurfacePermissionSet::new(["a"], ["x"]);
 assert!(set .derive_capability_subset(["a", "b"], None::<&str>) .is_err());
 assert!(set .derive_capability_subset(["a"], ["y"]) .is_err());
        let ok = set.derive_capability_subset(["a"], ["x"]).unwrap();
        assert!(ok.is_live());
        assert!(ok.allows_tool("a"));
    }
}
}


// --- inlined lenses/capsule.rs ---
pub mod capsule {
//! Typed handoff capsules: CLAIM, never CAPABILITY.
//!
//! Every transfer between surfaces creates a typed capsule, never a flattened
//! transcript. The single most important invariant:
//!
//! > A capsule never widens permission. Receiving a capsule in CHAT does not
//! > grant CHAT the connector access YOU held when it was created. The capsule
//! > carries the CLAIM, never the CAPABILITY.
//!
//! Enforced as a type boundary: there is no API that turns a capsule into a
//! [`SurfaceCapability`]. Attempts to extract capability fail closed and are
//! testable.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lenses::capability::{CapabilitySnapshot, SurfaceCapability};
use crate::lenses::error::{Result, YouError};
use crate::lenses::evidence::EvidenceTier;
use crate::lenses::surface::Surface;

/// Direction of a typed surface handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffKind {
    /// YOU → CHAT: implementation campaign.
    YouToChat,
    /// CHAT → IDE: repository/worktree and verification plan.
    ChatToIde,
    /// IDE → YOU: release summary.
    IdeToYou,
}

impl HandoffKind {
    pub fn from_surface(self) -> Surface {
        match self {
            Self::YouToChat => Surface::You,
            Self::ChatToIde => Surface::Chat,
            Self::IdeToYou => Surface::Ide,
        }
    }

    pub fn to_surface(self) -> Surface {
        match self {
            Self::YouToChat => Surface::Chat,
            Self::ChatToIde => Surface::Ide,
            Self::IdeToYou => Surface::You,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::YouToChat => "you_to_chat",
            Self::ChatToIde => "chat_to_ide",
            Self::IdeToYou => "ide_to_you",
        }
    }
}

/// One link in the provenance chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceEntry {
    pub actor: String,
    pub surface: Surface,
    pub at_ms: u64,
    pub action: String,
}

/// A claim the capsule asserts, with its evidence tier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub id: String,
    pub text: String,
    pub evidence_tier: EvidenceTier,
    /// Optional structured payload (goals, plans, summaries — never capabilities).
    #[serde(default)]
    pub payload: Value,
}

/// What the capsule deliberately leaves out, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliberateExclusion {
    pub item: String,
    pub reason: String,
}

/// Snapshot of permissions under which the capsule was created.
/// This is an audit claim, not a grant handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionSnapshot {
    pub surface: Surface,
    pub tools: Vec<String>,
    pub connectors: Vec<String>,
}

impl PermissionSnapshot {
    pub fn from_capability(surface: Surface, cap: &SurfaceCapability) -> Self {
        let snap = cap.snapshot();
        Self {
            surface,
            tools: snap.tools,
            connectors: snap.connectors,
        }
    }

    pub fn from_snapshot(surface: Surface, snap: &CapabilitySnapshot) -> Self {
        Self {
            surface,
            tools: snap.tools.clone(),
            connectors: snap.connectors.clone(),
        }
    }
}

/// Typed handoff capsule. Carries claims + audit metadata. **Does not grant
/// capability.** Fields that describe permissions are snapshots for provenance
/// only; the only way to obtain a live [`SurfaceCapability`] is surface-default
/// derivation (or an explicit session grant elsewhere), never from this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandoffCapsule {
    pub id: String,
    pub kind: HandoffKind,
    pub origin_surface: Surface,
    pub origin_session: String,
    pub target_surface: Surface,
    pub created_ms: u64,
    pub provenance: Vec<ProvenanceEntry>,
    pub claims: Vec<Claim>,
    /// Permissions under which this capsule was created (audit only).
    pub permissions_at_creation: PermissionSnapshot,
    pub deliberately_excludes: Vec<DeliberateExclusion>,
    /// blake3 hex of canonical claim+exclusion content.
    pub content_hash: String,
    /// Kind-specific body (campaign / verification plan / release summary).
    pub body: Value,
}

impl HandoffCapsule {
    /// Build a capsule. Computes content hash over claims + exclusions + body.
    pub fn seal(
        kind: HandoffKind,
        origin_session: impl Into<String>,
        created_ms: u64,
        provenance: Vec<ProvenanceEntry>,
        claims: Vec<Claim>,
        permissions_at_creation: PermissionSnapshot,
        deliberately_excludes: Vec<DeliberateExclusion>,
        body: Value,
    ) -> Result<Self> {
        let origin = kind.from_surface();
        let target = kind.to_surface();
        if permissions_at_creation.surface != origin {
            return Err(YouError::InvalidHandoff(format!(
                "permissions snapshot surface {:?} does not match origin {:?}",
                permissions_at_creation.surface, origin
            )));
        }
        let content_hash = hash_capsule_content(&claims, &deliberately_excludes, &body);
        Ok(Self {
            id: format!("hcap_{}", mint_ulid()),
            kind,
            origin_surface: origin,
            origin_session: origin_session.into(),
            target_surface: target,
            created_ms,
            provenance,
            claims,
            permissions_at_creation,
            deliberately_excludes,
            content_hash,
            body,
        })
    }

    /// Open for a receiving surface: returns claims and body only.
    /// Does **not** transfer or grant any capability from
    /// `permissions_at_creation`.
    pub fn open_for(&self, receiver: Surface) -> Result<OpenedCapsule> {
        if receiver != self.target_surface {
            return Err(YouError::InvalidHandoff(format!(
                "capsule targets {:?}, cannot open as {:?}",
                self.target_surface, receiver
            )));
        }
        Ok(OpenedCapsule {
            capsule_id: self.id.clone(),
            kind: self.kind,
            origin_surface: self.origin_surface,
            origin_session: self.origin_session.clone(),
            claims: self.claims.clone(),
            deliberately_excludes: self.deliberately_excludes.clone(),
            body: self.body.clone(),
            content_hash: self.content_hash.clone(),
            permissions_described: self.permissions_at_creation.clone(),
        })
    }

    /// Type-boundary enforcement: a capsule cannot be turned into a live
    /// capability. Always fails closed. Tests assert this path.
    pub fn try_extract_capability(&self) -> Result<SurfaceCapability> {
        Err(YouError::PolicyDenied(format!(
            "capsule {} carries claims only; cannot extract capability \
             (connectors at creation: {:?})",
            self.id, self.permissions_at_creation.connectors
        )))
    }

    /// Attempt to use a connector named in the creation snapshot as if the
    /// capsule granted it. Always fails — documents the attack surface.
    pub fn try_use_creator_connector(&self, connector: &str) -> Result<()> {
        if self
            .permissions_at_creation
            .connectors
            .iter()
            .any(|c| c == connector)
        {
            return Err(YouError::PolicyDenied(format!(
                "capsule describes creator connector '{connector}' but does not grant it; \
                 receiving surface must derive its own capability"
            )));
        }
        Err(YouError::CapabilityMissing(format!(
            "connector '{connector}' not even described on capsule"
        )))
    }

    /// Verify content hash integrity.
    pub fn verify_hash(&self) -> bool {
        let expected = hash_capsule_content(&self.claims, &self.deliberately_excludes, &self.body);
        expected == self.content_hash
    }
}

/// What a receiving surface gets after opening a capsule: claims + body +
/// audit description of creator permissions. No live capability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenedCapsule {
    pub capsule_id: String,
    pub kind: HandoffKind,
    pub origin_surface: Surface,
    pub origin_session: String,
    pub claims: Vec<Claim>,
    pub deliberately_excludes: Vec<DeliberateExclusion>,
    pub body: Value,
    pub content_hash: String,
    /// Audit-only description of what the origin held. Not a grant.
    pub permissions_described: PermissionSnapshot,
}

impl OpenedCapsule {
    /// There is no method on OpenedCapsule that yields SurfaceCapability.
    /// This helper makes the invariant explicit for callers and tests.
    pub fn grants_capability(&self) -> bool {
        false
    }
}

/// Result of a surface receiving a capsule into its session: claims are
/// ingested; the session's live capability is **unchanged**.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceivedHandoff {
    pub opened: OpenedCapsule,
    /// Capability the receiver held before and after receive (must be equal).
    pub receiver_capability_before: CapabilitySnapshot,
    pub receiver_capability_after: CapabilitySnapshot,
}

impl ReceivedHandoff {
    pub fn capability_unchanged(&self) -> bool {
        self.receiver_capability_before == self.receiver_capability_after
    }
}

/// A live surface session: holds its own derived capability. Receiving a
/// capsule never mutates that capability.
#[derive(Debug, Clone)]
pub struct SurfaceSession {
    pub surface: Surface,
    pub session_id: String,
    capability: SurfaceCapability,
}

impl SurfaceSession {
    pub fn open(surface: Surface, session_id: impl Into<String>) -> Self {
        let defaults = crate::lenses::surface::SurfaceDefaults::for_surface(surface);
        Self {
            surface,
            session_id: session_id.into(),
            capability: defaults.permissions.derive_capability(),
        }
    }

    pub fn with_capability(
        surface: Surface,
        session_id: impl Into<String>,
        capability: SurfaceCapability,
    ) -> Self {
        Self {
            surface,
            session_id: session_id.into(),
            capability,
        }
    }

    pub fn capability(&self) -> &SurfaceCapability {
        &self.capability
    }

    /// Receive a typed capsule. Ingests claims; **does not** widen capability
    /// to include creator connectors/tools.
    pub fn receive(&self, capsule: &HandoffCapsule) -> Result<ReceivedHandoff> {
        let before = self.capability.snapshot();
        let opened = capsule.open_for(self.surface)?;
        // Deliberately do nothing to self.capability — receive is &self.
        let after = self.capability.snapshot();
        Ok(ReceivedHandoff {
            opened,
            receiver_capability_before: before,
            receiver_capability_after: after,
        })
    }

    /// Gate a tool under this session's capability (not the capsule's snapshot).
    pub fn require_tool(&self, tool: &str) -> Result<()> {
        self.capability.require_tool(tool)
    }

    pub fn require_connector(&self, connector: &str) -> Result<()> {
        self.capability.require_connector(connector)
    }
}

fn hash_capsule_content(
    claims: &[Claim],
    exclusions: &[DeliberateExclusion],
    body: &Value,
) -> String {
    let payload = serde_json::json!({
        "claims": claims,
        "exclusions": exclusions,
        "body": body,
    });
    let bytes = serde_json::to_vec(&payload).unwrap_or_default();
    format!("blake3:{}", blake3::hash(&bytes).to_hex())
}

fn mint_ulid() -> String {
    ulid::Ulid::new().to_string().to_ascii_lowercase()
}
}


// --- inlined lenses/error.rs ---
pub mod error {
//! Error types for hide-you.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum YouError {
    #[error("policy denied: {0}")]
    PolicyDenied(String),

    #[error("capability missing: {0}")]
    CapabilityMissing(String),

    #[error("invalid handoff: {0}")]
    InvalidHandoff(String),

    #[error("budget exhausted: {0}")]
    BudgetExhausted(String),

    #[error("invalid state: {0}")]
    InvalidState(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("promotion refused: {0}")]
    PromotionRefused(String),

    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, YouError>;
}


// --- inlined lenses/evidence.rs ---
pub mod evidence {
//! Evidence tiers for claims carried in capsules and swarm conclusions.
//!
//! Higher tiers outrank lower ones. Consensus (votes) is not a tier; a
//! reproduced defect is stronger evidence than agreement among agents.

use serde::{Deserialize, Serialize};

/// How strongly a claim is supported. Ordered by authority (weakest first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceTier {
    /// Bare assertion with no backing.
    Asserted,
    /// Multiple agents agreed; weak — never sufficient alone for high-risk promotion.
    Consensus,
    /// Grounded in cited sources or prior research with quality grades.
    Cited,
    /// Independently checked by a distinct verifier agent.
    IndependentlyVerified,
    /// A defect or acceptance condition was reproduced (oracles, tests, fixtures).
    Reproduced,
}

impl EvidenceTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Asserted => "asserted",
            Self::Consensus => "consensus",
            Self::Cited => "cited",
            Self::IndependentlyVerified => "independently_verified",
            Self::Reproduced => "reproduced",
        }
    }

    /// True if this tier is strong enough to support promoting a high-risk
    /// conclusion (independent verification or reproduction).
    pub fn supports_high_risk_promotion(self) -> bool {
        matches!(self, Self::IndependentlyVerified | Self::Reproduced)
    }

    /// Consensus is deliberately weaker than reproduction.
    pub fn outranked_by_reproduction(self) -> bool {
        self < Self::Reproduced
    }
}

impl std::fmt::Display for EvidenceTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
}


// --- inlined lenses/fixture.rs ---
pub mod fixture {
//! Fixture model provider — no real inference.
//!
//! Deterministic canned replies keyed by role + goal fragment. Used so swarm
//! orchestration and handoff tests never load a model or touch Metal.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::lenses::agent::{AgentReceipt, AgentSpec};
use crate::lenses::capsule::Claim;
use crate::lenses::evidence::EvidenceTier;
use crate::lenses::roles::AgentRole;

/// One canned reply the fixture provider may return.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureReply {
    pub summary: String,
    pub tokens_used: u64,
    pub steps_used: u32,
    pub cpu_ms: u64,
    pub ram_mb: u64,
    pub evidence_tier: EvidenceTier,
    pub claim_texts: Vec<String>,
}

impl Default for FixtureReply {
    fn default() -> Self {
        Self {
            summary: "fixture ok".into(),
            tokens_used: 10,
            steps_used: 1,
            cpu_ms: 5,
            ram_mb: 16,
            evidence_tier: EvidenceTier::Asserted,
            claim_texts: vec!["fixture claim".into()],
        }
    }
}

/// Model-free provider. Role-keyed defaults; optional overrides by agent id.
#[derive(Debug, Clone, Default)]
pub struct FixtureProvider {
    role_defaults: std::collections::BTreeMap<String, FixtureReply>,
    agent_overrides: std::collections::BTreeMap<String, FixtureReply>,
}

impl FixtureProvider {
    pub fn new() -> Self {
        let mut p = Self::default();
        for role in AgentRole::all() {
            p.role_defaults.insert(
                role.as_str().to_string(),
                FixtureReply {
                    summary: format!("fixture:{role}"),
                    tokens_used: 12,
                    steps_used: 1,
                    cpu_ms: 8,
                    ram_mb: 20,
                    evidence_tier: match role {
                        AgentRole::Verifier | AgentRole::FactChecker => {
                            EvidenceTier::IndependentlyVerified
                        }
                        AgentRole::Researcher => EvidenceTier::Cited,
                        _ => EvidenceTier::Asserted,
                    },
                    claim_texts: vec![format!("{role} output")],
                },
            );
        }
        p
    }

    pub fn override_agent(mut self, agent_id: &str, reply: FixtureReply) -> Self {
        self.agent_overrides.insert(agent_id.to_string(), reply);
        self
    }

    pub fn override_role(mut self, role: AgentRole, reply: FixtureReply) -> Self {
        self.role_defaults.insert(role.as_str().to_string(), reply);
        self
    }

    /// Produce a deterministic receipt for an agent. No model call.
    pub fn run(&self, spec: &AgentSpec) -> AgentReceipt {
        let reply = self
            .agent_overrides
            .get(spec.id.as_str())
            .or_else(|| self.role_defaults.get(spec.role.as_str()))
            .cloned()
            .unwrap_or_default();

        let claims: Vec<Claim> = reply
            .claim_texts
            .iter()
            .enumerate()
            .map(|(i, text)| Claim {
                id: format!("clm_{}_{}", spec.id.as_str(), i),
                text: text.clone(),
                evidence_tier: reply.evidence_tier,
                payload: json!({
                    "goal": spec.goal,
                    "role": spec.role.as_str(),
                    "model_profile": spec.model_profile,
                }),
            })
            .collect();

        AgentReceipt {
            agent_id: spec.id.clone(),
            role: spec.role,
            ok: true,
            summary: reply.summary,
            tokens_used: reply.tokens_used,
            steps_used: reply.steps_used,
            cpu_ms: reply.cpu_ms,
            ram_mb: reply.ram_mb,
            claims,
            evidence_tier: reply.evidence_tier,
        }
    }

    /// Inspectable catalog for contracts/tests.
    pub fn catalog(&self) -> Value {
        json!({
            "kind": "fixture_provider",
            "roles": self.role_defaults.keys().cloned().collect::<Vec<_>>(),
            "agent_overrides": self.agent_overrides.keys().cloned().collect::<Vec<_>>(),
            "real_inference": false,
        })
    }
}
}


// --- inlined lenses/modes.rs ---
pub mod modes {
//! Swarm operating modes.

use serde::{Deserialize, Serialize};

/// How a swarm coordinates its agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmMode {
    ParallelResearch,
    Debate,
    IndependentReplication,
    PlanTournament,
    CreativeDivergence,
    FactVerification,
    DocumentAssembly,
    PersonalAdministration,
}

impl SwarmMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ParallelResearch => "parallel_research",
            Self::Debate => "debate",
            Self::IndependentReplication => "independent_replication",
            Self::PlanTournament => "plan_tournament",
            Self::CreativeDivergence => "creative_divergence",
            Self::FactVerification => "fact_verification",
            Self::DocumentAssembly => "document_assembly",
            Self::PersonalAdministration => "personal_administration",
        }
    }

    pub fn all() -> &'static [SwarmMode] {
        &[
            Self::ParallelResearch,
            Self::Debate,
            Self::IndependentReplication,
            Self::PlanTournament,
            Self::CreativeDivergence,
            Self::FactVerification,
            Self::DocumentAssembly,
            Self::PersonalAdministration,
        ]
    }
}

impl std::fmt::Display for SwarmMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
}


// --- inlined lenses/project.rs ---
pub mod project {
//! Projects: unify conversations, documents, objects, connectors, plans,
//! tasks, memory, automations, agents, and artifacts under lifecycle states.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Stable project id (`prj_…`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProjectId(pub String);

impl ProjectId {
    pub fn new() -> Self {
        Self(format!(
            "prj_{}",
            ulid::Ulid::new().to_string().to_ascii_lowercase()
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ProjectId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ProjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Project lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectState {
    Explore,
    Plan,
    Execute,
    Review,
    Archive,
}

impl ProjectState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Plan => "plan",
            Self::Execute => "execute",
            Self::Review => "review",
            Self::Archive => "archive",
        }
    }

    pub fn all() -> &'static [ProjectState] {
        &[
            Self::Explore,
            Self::Plan,
            Self::Execute,
            Self::Review,
            Self::Archive,
        ]
    }

    /// Legal transitions (forward + archive from any non-archive).
    pub fn can_transition_to(self, next: ProjectState) -> bool {
        use ProjectState::*;
        if self == next {
            return true;
        }
        if next == Archive {
            return self != Archive;
        }
        matches!(
            (self, next),
            (Explore, Plan)
                | (Plan, Explore)
                | (Plan, Execute)
                | (Execute, Plan)
                | (Execute, Review)
                | (Review, Execute)
                | (Review, Archive)
                | (Explore, Execute) // skip plan when work is obvious
        )
    }
}

/// Kinds of members a project unifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectMemberKind {
    Conversation,
    Document,
    Object,
    Connector,
    Plan,
    Task,
    Memory,
    Automation,
    Agent,
    Artifact,
}

impl ProjectMemberKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conversation => "conversation",
            Self::Document => "document",
            Self::Object => "object",
            Self::Connector => "connector",
            Self::Plan => "plan",
            Self::Task => "task",
            Self::Memory => "memory",
            Self::Automation => "automation",
            Self::Agent => "agent",
            Self::Artifact => "artifact",
        }
    }

    pub fn all() -> &'static [ProjectMemberKind] {
        &[
            Self::Conversation,
            Self::Document,
            Self::Object,
            Self::Connector,
            Self::Plan,
            Self::Task,
            Self::Memory,
            Self::Automation,
            Self::Agent,
            Self::Artifact,
        ]
    }
}

/// A reference to something the project unifies (by kind + external id).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectMember {
    pub kind: ProjectMemberKind,
    pub ref_id: String,
    #[serde(default)]
    pub label: Option<String>,
}

/// Unified project container.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub id: ProjectId,
    pub name: String,
    pub state: ProjectState,
    pub members: Vec<ProjectMember>,
    /// Free-form notes / goal for the project.
    pub summary: String,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// Optional link to an active swarm.
    pub active_swarm_id: Option<String>,
    /// Metadata bag (session ids, surface tags, etc.).
    #[serde(default)]
    pub meta: BTreeMap<String, Value>,
}

impl Project {
    pub fn create(name: impl Into<String>, summary: impl Into<String>, now_ms: u64) -> Self {
        Self {
            id: ProjectId::new(),
            name: name.into(),
            state: ProjectState::Explore,
            members: Vec::new(),
            summary: summary.into(),
            created_ms: now_ms,
            updated_ms: now_ms,
            active_swarm_id: None,
            meta: BTreeMap::new(),
        }
    }

    pub fn transition(&mut self, next: ProjectState, now_ms: u64) -> crate::lenses::Result<()> {
        if !self.state.can_transition_to(next) {
            return Err(crate::lenses::YouError::InvalidState(format!(
                "cannot transition project {} from {:?} to {:?}",
                self.id, self.state, next
            )));
        }
        self.state = next;
        self.updated_ms = now_ms;
        Ok(())
    }

    pub fn attach(
        &mut self,
        kind: ProjectMemberKind,
        ref_id: impl Into<String>,
        label: Option<String>,
        now_ms: u64,
    ) {
        let ref_id = ref_id.into();
        if self
            .members
            .iter()
            .any(|m| m.kind == kind && m.ref_id == ref_id)
        {
            return;
        }
        self.members.push(ProjectMember {
            kind,
            ref_id,
            label,
        });
        self.updated_ms = now_ms;
    }

    pub fn members_of(&self, kind: ProjectMemberKind) -> impl Iterator<Item = &ProjectMember> {
        self.members.iter().filter(move |m| m.kind == kind)
    }

    pub fn link_swarm(&mut self, swarm_id: impl Into<String>, now_ms: u64) {
        self.active_swarm_id = Some(swarm_id.into());
        self.updated_ms = now_ms;
    }

    pub fn declaration(&self) -> Value {
        serde_json::json!({
            "id": self.id.as_str(),
            "name": self.name,
            "state": self.state.as_str(),
            "summary": self.summary,
            "member_counts": ProjectMemberKind::all().iter().map(|k| {
                (k.as_str(), self.members_of(*k).count())
            }).collect::<BTreeMap<_, _>>(),
            "members": self.members,
            "active_swarm_id": self.active_swarm_id,
            "created_ms": self.created_ms,
            "updated_ms": self.updated_ms,
        })
    }
}
}


// --- inlined lenses/promotion.rs ---
pub mod promotion {
//! Independent verification for high-risk conclusion promotion.
//!
//! Law: **no agent promotes its own high-risk conclusion.** Promotion needs
//! independent verification. Consensus is weak evidence; a reproduced defect
//! outranks votes.

use serde::{Deserialize, Serialize};

use crate::lenses::agent::AgentId;
use crate::lenses::error::{Result, YouError};
use crate::lenses::evidence::EvidenceTier;
use crate::lenses::roles::AgentRole;

/// Risk class of a conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConclusionRisk {
    Low,
    High,
}

/// A swarm conclusion awaiting (or denied) promotion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conclusion {
    pub id: String,
    pub text: String,
    /// Authoring agent — cannot self-promote when risk is High.
    pub author_agent_id: AgentId,
    pub author_role: AgentRole,
    pub risk: ConclusionRisk,
    pub evidence_tier: EvidenceTier,
}

/// Evidence presented to the promotion board.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum PromotionEvidence {
    /// Votes from other agents. Weak; never sufficient alone for high-risk.
    Consensus { tally: VoteTally },
    /// Distinct verifier agent confirms.
    IndependentVerification {
        verifier_agent_id: AgentId,
        verifier_role: AgentRole,
        note: String,
    },
    /// A defect/oracle was reproduced. Strongest common path.
    Reproduction {
        reproducer_agent_id: AgentId,
        detail: String,
    },
}

/// Vote counts (weak evidence).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct VoteTally {
    pub for_promotion: u32,
    pub against: u32,
    pub abstain: u32,
}

impl VoteTally {
    pub fn majority_for(&self) -> bool {
        self.for_promotion > self.against && self.for_promotion > 0
    }
}

/// Outcome of a promotion attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum PromotionDecision {
    Promoted {
        conclusion_id: String,
        basis: String,
        evidence_tier: EvidenceTier,
    },
    Refused {
        conclusion_id: String,
        reason: String,
    },
}

/// Board that enforces independent verification for high-risk promotions.
#[derive(Debug, Clone, Default)]
pub struct PromotionBoard {
    pub decisions: Vec<PromotionDecision>,
}

impl PromotionBoard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attempt to promote `conclusion` with the given evidence.
    ///
    /// Rules:
    /// 1. Low risk may promote on cited+ evidence without independent verifier.
    /// 2. High risk requires independent verification or reproduction.
    /// 3. Author cannot be the sole verifier (self-promotion banned).
    /// 4. Consensus alone never promotes high-risk, even with unanimous votes.
    /// 5. Reproduction outranks consensus when both are present.
    pub fn try_promote(
        &mut self,
        conclusion: &Conclusion,
        evidence: &[PromotionEvidence],
    ) -> Result<PromotionDecision> {
        let decision = match conclusion.risk {
            ConclusionRisk::Low => self.promote_low(conclusion, evidence),
            ConclusionRisk::High => self.promote_high(conclusion, evidence),
        };
        self.decisions.push(decision.clone());
        match &decision {
            PromotionDecision::Promoted { .. } => Ok(decision),
            PromotionDecision::Refused { reason, .. } => {
                Err(YouError::PromotionRefused(reason.clone()))
            }
        }
    }

    fn promote_low(
        &self,
        conclusion: &Conclusion,
        evidence: &[PromotionEvidence],
    ) -> PromotionDecision {
        if evidence.is_empty() && conclusion.evidence_tier < EvidenceTier::Cited {
            return PromotionDecision::Refused {
                conclusion_id: conclusion.id.clone(),
                reason: "low-risk promotion still needs cited+ tier or supporting evidence".into(),
            };
        }
        PromotionDecision::Promoted {
            conclusion_id: conclusion.id.clone(),
            basis: "low_risk".into(),
            evidence_tier: conclusion.evidence_tier,
        }
    }

    fn promote_high(
        &self,
        conclusion: &Conclusion,
        evidence: &[PromotionEvidence],
    ) -> PromotionDecision {
        // Scan for forbidden self-promotion first.
        for e in evidence {
            if let PromotionEvidence::IndependentVerification {
                verifier_agent_id,
                ..
            } = e
            {
                if verifier_agent_id == &conclusion.author_agent_id {
                    return PromotionDecision::Refused {
                        conclusion_id: conclusion.id.clone(),
                        reason: format!(
                            "agent {} cannot promote its own high-risk conclusion",
                            conclusion.author_agent_id
                        ),
                    };
                }
            }
            if let PromotionEvidence::Reproduction {
                reproducer_agent_id,
                ..
            } = e
            {
                if reproducer_agent_id == &conclusion.author_agent_id {
                    return PromotionDecision::Refused {
                        conclusion_id: conclusion.id.clone(),
                        reason: format!(
                            "agent {} cannot self-reproduce to promote its own high-risk conclusion",
                            conclusion.author_agent_id
                        ),
                    };
                }
            }
        }

        // Reproduction outranks everything else when present and independent.
        if let Some(PromotionEvidence::Reproduction { detail, .. }) = evidence.iter().find(|e| {
            matches!(e, PromotionEvidence::Reproduction { reproducer_agent_id, .. }
                if reproducer_agent_id != &conclusion.author_agent_id)
        }) {
            return PromotionDecision::Promoted {
                conclusion_id: conclusion.id.clone(),
                basis: format!("reproduction:{detail}"),
                evidence_tier: EvidenceTier::Reproduced,
            };
        }

        // Independent verification by a Verifier (or any non-author agent).
        if let Some(PromotionEvidence::IndependentVerification {
            verifier_agent_id,
            verifier_role,
            note,
        }) = evidence.iter().find(|e| {
            matches!(e, PromotionEvidence::IndependentVerification { verifier_agent_id, .. }
                if verifier_agent_id != &conclusion.author_agent_id)
        }) {
            // Prefer Verifier role but any distinct agent counts as independent.
            let _ = verifier_role;
            return PromotionDecision::Promoted {
                conclusion_id: conclusion.id.clone(),
                basis: format!("independent_verification:{verifier_agent_id}:{note}"),
                evidence_tier: EvidenceTier::IndependentlyVerified,
            };
        }

        // Consensus alone is weak — refuse high-risk even with majority.
        if evidence
            .iter()
            .any(|e| matches!(e, PromotionEvidence::Consensus { tally } if tally.majority_for()))
        {
            return PromotionDecision::Refused {
                conclusion_id: conclusion.id.clone(),
                reason: "consensus is weak evidence; high-risk promotion requires independent \
                         verification or reproduction"
                    .into(),
            };
        }

        PromotionDecision::Refused {
            conclusion_id: conclusion.id.clone(),
            reason: "high-risk conclusion lacks independent verification or reproduction".into(),
        }
    }
}
}


// --- inlined lenses/roles.rs ---
pub mod roles {
//! Swarm agent roles — governed team positions, not prompt labels alone.

use serde::{Deserialize, Serialize};

/// Role each swarm agent holds. A YOU swarm is a governed team.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    Researcher,
    Planner,
    Critic,
    FactChecker,
    Writer,
    DataAnalyst,
    ImageAnalyst,
    ConnectorOperator,
    Scheduler,
    Archivist,
    Specialist,
    /// Independent check path; never the author of a conclusion it promotes.
    Verifier,
}

impl AgentRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Researcher => "researcher",
            Self::Planner => "planner",
            Self::Critic => "critic",
            Self::FactChecker => "fact_checker",
            Self::Writer => "writer",
            Self::DataAnalyst => "data_analyst",
            Self::ImageAnalyst => "image_analyst",
            Self::ConnectorOperator => "connector_operator",
            Self::Scheduler => "scheduler",
            Self::Archivist => "archivist",
            Self::Specialist => "specialist",
            Self::Verifier => "verifier",
        }
    }

    pub fn all() -> &'static [AgentRole] {
        &[
            Self::Researcher,
            Self::Planner,
            Self::Critic,
            Self::FactChecker,
            Self::Writer,
            Self::DataAnalyst,
            Self::ImageAnalyst,
            Self::ConnectorOperator,
            Self::Scheduler,
            Self::Archivist,
            Self::Specialist,
            Self::Verifier,
        ]
    }

    /// Verifier may confirm others' conclusions; it does not author promotions
    /// of its own high-risk claims without a second independent path.
    pub fn is_verifier(self) -> bool {
        matches!(self, Self::Verifier)
    }
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
}


// --- inlined lenses/session_graph.rs ---
pub mod session_graph {
//! One HIDE session, three surface lenses.
//!
//! Doctrine (`HIDE_YOU_SURFACE_AUTHORITY.json`): YOU, CHAT and IDE are three
//! **lenses** over one session. They share session identity and must not each
//! own a copy of memory, objects, connectors, or the event stream. What differs
//! is default context and default **capability** (non-widening, derived once).
//!
//! A handoff capsule still carries a CLAIM only. Switching the active lens, or
//! receiving a capsule into a lens, never transports authority.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lenses::capsule::{
    Claim, DeliberateExclusion, HandoffCapsule, HandoffKind, OpenedCapsule, PermissionSnapshot,
    ProvenanceEntry, ReceivedHandoff, SurfaceSession,
};
use crate::lenses::error::{Result, YouError};
use crate::lenses::surface::Surface;

/// Shared product session: one identity, three capability lenses, typed handoffs.
///
/// Surfaces do not construct independent sessions. They call [`SurfaceGraph::lens`]
/// and [`SurfaceGraph::switch`]. The host owns the single durable event log /
/// memory / object store; this graph only holds the surface authority view.
#[derive(Debug, Clone)]
pub struct SurfaceGraph {
    session_id: String,
    active: Surface,
    lenses: BTreeMap<Surface, SurfaceSession>,
    /// Sealed outbound capsules, keyed by capsule id.
    capsules: BTreeMap<String, HandoffCapsule>,
    /// Claims received per surface (capability never stored here).
    inbox: BTreeMap<Surface, Vec<OpenedCapsule>>,
}

/// Read-only snapshot a FE / projection can render without holding live capability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceGraphView {
    pub session_id: String,
    pub active_surface: String,
    /// Per-surface audit capability description (tools + connectors). Not a grant handle.
    pub lenses: BTreeMap<String, LensView>,
    pub unread_handoffs: usize,
    pub capsules: Vec<CapsuleView>,
    pub inbox: BTreeMap<String, Vec<OpenedCapsuleView>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LensView {
    pub surface: String,
    pub tools: Vec<String>,
    pub connectors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapsuleView {
    pub id: String,
    pub kind: String,
    pub origin_surface: String,
    pub target_surface: String,
    pub content_hash: String,
    pub claim_count: usize,
    pub exclusion_count: usize,
    pub exclusions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenedCapsuleView {
    pub capsule_id: String,
    pub kind: String,
    pub origin_surface: String,
    pub claim_count: usize,
    pub content_hash: String,
    /// Audit-only description of creator permissions. Not a grant.
    pub permissions_described_tools: Vec<String>,
    pub permissions_described_connectors: Vec<String>,
}

impl SurfaceGraph {
    /// Open a graph on one session id. All three lenses share that id and hold
    /// surface-default capabilities only.
    pub fn open(session_id: impl Into<String>) -> Self {
        let session_id = session_id.into();
        let mut lenses = BTreeMap::new();
        for surface in Surface::all() {
            lenses.insert(
                surface,
                SurfaceSession::open(surface, session_id.clone()),
            );
        }
        let mut inbox = BTreeMap::new();
        for surface in Surface::all() {
            inbox.insert(surface, Vec::new());
        }
        Self {
            session_id,
            // Doctrine: Workstation / Chat is the front door; YOU is a lens, not a silo.
            active: Surface::Chat,
            lenses,
            capsules: BTreeMap::new(),
            inbox,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn active(&self) -> Surface {
        self.active
    }

    /// Switch the active lens. Does not mint a new session. Does not change any
    /// surface's capability.
    pub fn switch(&mut self, surface: Surface) -> Surface {
        self.active = surface;
        self.active
    }

    /// Borrow the live lens for a surface. All lenses share [`Self::session_id`].
    pub fn lens(&self, surface: Surface) -> Result<&SurfaceSession> {
        self.lenses
            .get(&surface)
            .ok_or_else(|| YouError::InvalidState(format!("missing lens for {surface}")))
    }

    pub fn active_lens(&self) -> Result<&SurfaceSession> {
        self.lens(self.active)
    }

    /// Seal a typed handoff from the active surface (or an explicit origin).
    ///
    /// The capsule records a permission **snapshot** for audit only. Live
    /// capability stays on the origin lens; the capsule cannot reconstitute it.
    pub fn create_handoff(
        &mut self,
        kind: HandoffKind,
        created_ms: u64,
        claims: Vec<Claim>,
        deliberately_excludes: Vec<DeliberateExclusion>,
        body: Value,
        actor: impl Into<String>,
    ) -> Result<HandoffCapsule> {
        let origin = kind.from_surface();
        let origin_lens = self.lens(origin)?;
        // Caller may only create handoffs from a surface that is part of this graph
        // and that matches the kind's origin. Active surface should be the origin
        // (prevents CHAT sealing a YOU→CHAT capsule while pretending to be YOU).
        if self.active != origin {
            return Err(YouError::PolicyDenied(format!(
                "active surface is {}; handoff kind {} requires origin {}",
                self.active,
                kind.as_str(),
                origin
            )));
        }
        let permissions =
            PermissionSnapshot::from_capability(origin, origin_lens.capability());
        let provenance = vec![ProvenanceEntry {
            actor: actor.into(),
            surface: origin,
            at_ms: created_ms,
            action: format!("handoff_{}", kind.as_str()),
        }];
        // Shared session identity on every capsule (one session, three lenses).
        let capsule = HandoffCapsule::seal(
            kind,
            self.session_id.clone(),
            created_ms,
            provenance,
            claims,
            permissions,
            deliberately_excludes,
            body,
        )?;
        self.capsules
            .insert(capsule.id.clone(), capsule.clone());
        Ok(capsule)
    }

    /// Receive a sealed capsule into its target lens on **this same session**.
    ///
    /// Capability of the target lens is unchanged. Creator connectors remain
    /// unusable on the receiver. Claims land in the target inbox.
    pub fn receive_handoff(&mut self, capsule_id: &str) -> Result<ReceivedHandoff> {
        let capsule = self
            .capsules
            .get(capsule_id)
            .cloned()
            .ok_or_else(|| {
                YouError::InvalidHandoff(format!("unknown capsule id {capsule_id}"))
            })?;
        // Capsules created elsewhere for a different session are refused: lenses
        // share one session, not a free-floating claim bus across sessions.
        if capsule.origin_session != self.session_id {
            return Err(YouError::InvalidHandoff(format!(
                "capsule session {} does not match graph session {}",
                capsule.origin_session, self.session_id
            )));
        }
        let target = capsule.target_surface;
        let lens = self.lens(target)?;
        let received = lens.receive(&capsule)?;
        // Type boundary re-check at the graph boundary (defense in depth).
        if !received.capability_unchanged() {
            return Err(YouError::PolicyDenied(
                "receive mutated receiver capability; refused".into(),
            ));
        }
        if let Err(err) = capsule.try_extract_capability() {
            // Expected always. Keep the error path live so a regression that
            // starts succeeding is not silently ignored.
            let _ = err;
        } else {
            return Err(YouError::PolicyDenied(
                "capsule yielded a capability; refused".into(),
            ));
        }
        self.inbox
            .entry(target)
            .or_default()
            .push(received.opened.clone());
        Ok(received)
    }

    /// Import a capsule sealed outside this graph (e.g. restored from the event
    /// log). Still refuses if its origin_session disagrees with this session.
    pub fn admit_capsule(&mut self, capsule: HandoffCapsule) -> Result<()> {
        if capsule.origin_session != self.session_id {
            return Err(YouError::InvalidHandoff(format!(
                "admit refused: capsule session {} != graph {}",
                capsule.origin_session, self.session_id
            )));
        }
        // Never trust a capsule that could somehow extract capability.
        if capsule.try_extract_capability().is_ok() {
            return Err(YouError::PolicyDenied(
                "admit refused: capsule carries capability".into(),
            ));
        }
        self.capsules.insert(capsule.id.clone(), capsule);
        Ok(())
    }

    pub fn capsule(&self, id: &str) -> Option<&HandoffCapsule> {
        self.capsules.get(id)
    }

    pub fn inbox_for(&self, surface: Surface) -> &[OpenedCapsule] {
        self.inbox
            .get(&surface)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn unread_handoff_count(&self) -> usize {
        self.inbox.values().map(|v| v.len()).sum()
    }

    /// Projection for Wire-B / FE: state only, no live capability handles.
    pub fn view(&self) -> SurfaceGraphView {
        let mut lenses = BTreeMap::new();
        for surface in Surface::all() {
            if let Ok(lens) = self.lens(surface) {
                let snap = lens.capability().snapshot();
                lenses.insert(
                    surface.as_str().to_string(),
                    LensView {
                        surface: surface.as_str().to_string(),
                        tools: snap.tools,
                        connectors: snap.connectors,
                    },
                );
            }
        }
        let capsules: Vec<CapsuleView> = self
            .capsules
            .values()
            .map(|c| CapsuleView {
                id: c.id.clone(),
                kind: c.kind.as_str().to_string(),
                origin_surface: c.origin_surface.as_str().to_string(),
                target_surface: c.target_surface.as_str().to_string(),
                content_hash: c.content_hash.clone(),
                claim_count: c.claims.len(),
                exclusion_count: c.deliberately_excludes.len(),
                exclusions: c
                    .deliberately_excludes
                    .iter()
                    .map(|e| format!("{} ({})", e.item, e.reason))
                    .collect(),
            })
            .collect();
        let mut inbox = BTreeMap::new();
        for surface in Surface::all() {
            let items: Vec<OpenedCapsuleView> = self
                .inbox_for(surface)
                .iter()
                .map(|o| OpenedCapsuleView {
                    capsule_id: o.capsule_id.clone(),
                    kind: o.kind.as_str().to_string(),
                    origin_surface: o.origin_surface.as_str().to_string(),
                    claim_count: o.claims.len(),
                    content_hash: o.content_hash.clone(),
                    permissions_described_tools: o.permissions_described.tools.clone(),
                    permissions_described_connectors: o.permissions_described.connectors.clone(),
                })
                .collect();
            inbox.insert(surface.as_str().to_string(), items);
        }
        SurfaceGraphView {
            session_id: self.session_id.clone(),
            active_surface: self.active.as_str().to_string(),
            lenses,
            unread_handoffs: self.unread_handoff_count(),
            capsules,
            inbox,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lenses::evidence::EvidenceTier;
    use serde_json::json;
    #[test]
    fn three_lenses_share_one_session_id() {
        let g = SurfaceGraph::open("ses_shared");
        assert_eq!(g.session_id(), "ses_shared");
        for s in Surface::all() {
            assert_eq!(g.lens(s).unwrap().session_id, "ses_shared");
        }
 assert!(g .lens(Surface::You) .unwrap() .capability() .allows_connector("gmail"));
 assert!(!g .lens(Surface::Chat) .unwrap() .capability() .allows_connector("gmail"));
 assert!(!g .lens(Surface::Ide) .unwrap() .capability() .allows_connector("gmail"));
    }
    #[test]
    fn switch_does_not_change_session_or_capability() {
        let mut g = SurfaceGraph::open("ses_1");
        let before_chat = g.lens(Surface::Chat).unwrap().capability().snapshot();
        let before_you = g.lens(Surface::You).unwrap().capability().snapshot();
        g.switch(Surface::You);
        assert_eq!(g.active(), Surface::You);
        assert_eq!(g.session_id(), "ses_1");
 assert_eq!( g.lens(Surface::Chat).unwrap().capability().snapshot(), before_chat );
 assert_eq!( g.lens(Surface::You).unwrap().capability().snapshot(), before_you );
    }
    #[test]
    fn handoff_claim_never_grants_creator_capability_on_shared_session() {
        let mut g = SurfaceGraph::open("ses_shared");
        g.switch(Surface::You);
        let capsule = g
            .create_handoff(
                HandoffKind::YouToChat,
                1_000,
                vec![Claim {
                    id: "c1".into(),
                    text: "build triage worker".into(),
                    evidence_tier: EvidenceTier::Cited,
                    payload: json!({}),
                }],
                vec![DeliberateExclusion {
                    item: "gmail credentials".into(),
                    reason: "claim only".into(),
                }],
                json!({"kind": "implementation_campaign", "goal": "triage"}),
                "user",
            )
            .expect("create");
        assert_eq!(capsule.origin_session, "ses_shared");
        assert!(capsule.try_extract_capability().is_err());
        let received = g.receive_handoff(&capsule.id).expect("receive");
        assert!(received.capability_unchanged());
        assert!(!received.opened.grants_capability());
 assert!(g .lens(Surface::Chat) .unwrap() .require_connector("gmail") .is_err());
 assert!(g .lens(Surface::You) .unwrap() .require_connector("gmail") .is_ok());
        assert_eq!(g.inbox_for(Surface::Chat).len(), 1);
    }
    #[test]
    fn cannot_seal_handoff_from_wrong_active_surface() {
        let mut g = SurfaceGraph::open("ses_x");
        let err = g
            .create_handoff(
                HandoffKind::YouToChat,
                1,
                vec![],
                vec![],
                json!({}),
                "user",
            )
            .unwrap_err();
 assert!( err.to_string().contains("active surface"), "wrong origin refused: {err}" );
    }
}
}


// --- inlined lenses/surface.rs ---
pub mod surface {
//! The three HIDE surfaces (lenses over one session).

use serde::{Deserialize, Serialize};

use crate::lenses::capability::SurfacePermissionSet;

/// YOU / CHAT / IDE — three lenses, one session. They differ in default
/// context and default permissions, not in intelligence or truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// Private general-purpose multimodal personal AI.
    You,
    /// Repository-aware coding-agent workspace.
    Chat,
    /// Visual code and software-development environment.
    Ide,
}

impl Surface {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::You => "you",
            Self::Chat => "chat",
            Self::Ide => "ide",
        }
    }

    pub fn all() -> [Surface; 3] {
        [Self::You, Self::Chat, Self::Ide]
    }
}

impl std::fmt::Display for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Default permission profile for a surface, matching
/// `HIDE_YOU_SURFACE_AUTHORITY.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceDefaults {
    pub surface: Surface,
    /// Connector access description (metadata; closed set lives in permissions).
    pub connectors_policy: String,
    pub shell_policy: String,
    pub repo_write_policy: String,
    pub network_policy: String,
    /// Closed tool/connector set the surface may hold by default.
    pub permissions: SurfacePermissionSet,
}

impl SurfaceDefaults {
    /// YOU: connectors read-only, shell denied, repo write denied.
    pub fn you_default() -> Self {
        Self {
            surface: Surface::You,
            connectors_policy: "read-only".into(),
            shell_policy: "denied".into(),
            repo_write_policy: "denied".into(),
            network_policy: "explicit per session type".into(),
            // YOU may hold personal connectors (mail, calendar, vault) — read.
            permissions: SurfacePermissionSet::new(
                ["connector.read", "memory.read", "research.read", "object.read"],
                ["gmail", "calendar", "personal_vault", "rss"],
            ),
        }
    }

    /// CHAT: repo-scoped connector read, shell under policy, repo write via effects.
    pub fn chat_default() -> Self {
        Self {
            surface: Surface::Chat,
            connectors_policy: "repo-scoped read".into(),
            shell_policy: "under policy".into(),
            repo_write_policy: "via effects".into(),
            network_policy: "denied by default".into(),
            // CHAT deliberately does NOT include personal connectors.
            permissions: SurfacePermissionSet::new(
                [
                    "repo.read",
                    "repo.write_effect",
                    "shell.under_policy",
                    "object.read",
                ],
                ["repo_index"],
            ),
        }
    }

    /// IDE: repo-scoped read, shell under policy, repo write via effects + visible diff.
    pub fn ide_default() -> Self {
        Self {
            surface: Surface::Ide,
            connectors_policy: "repo-scoped read".into(),
            shell_policy: "under policy".into(),
            repo_write_policy: "via effects with visible diff".into(),
            network_policy: "denied by default".into(),
            permissions: SurfacePermissionSet::new(
                [
                    "repo.read",
                    "repo.write_effect",
                    "shell.under_policy",
                    "diff.present",
                    "object.read",
                ],
                ["repo_index", "source_control"],
            ),
        }
    }

    pub fn for_surface(surface: Surface) -> Self {
        match surface {
            Surface::You => Self::you_default(),
            Surface::Chat => Self::chat_default(),
            Surface::Ide => Self::ide_default(),
        }
    }
}
}


// --- inlined lenses/swarm.rs ---
pub mod swarm {
//! YOU swarms — governed teams on the fleet substrate concept.
//!
//! A swarm is not prompt multiplication: each agent receives goal, role,
//! context capsule, model/profile, tools/connectors, permissions, budget,
//! deadline, output schema, and verification contract. Resource economics
//! are enforced; exceeding budget halts the swarm and records why.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lenses::agent::{AgentReceipt, AgentSpec};
use crate::lenses::budget::{BudgetUsage, ResourceBudget, StopCondition, StopReason, SwarmBudget};
use crate::lenses::capability::{SurfaceCapability, SurfacePermissionSet};
use crate::lenses::error::{Result, YouError};
use crate::lenses::fixture::FixtureProvider;
use crate::lenses::modes::SwarmMode;
use crate::lenses::roles::AgentRole;

/// Stable swarm id (`swm_…`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SwarmId(pub String);

impl SwarmId {
    pub fn new() -> Self {
        Self(format!(
            "swm_{}",
            ulid::Ulid::new().to_string().to_ascii_lowercase()
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SwarmId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SwarmId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Lifecycle of a swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmStatus {
    Pending,
    Running,
    /// Budget or stop condition fired; further steps refuse.
    Halted,
    Completed,
    Cancelled,
}

/// A governed swarm: mode, agents, shared goal, budget, permission root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Swarm {
    pub id: SwarmId,
    pub goal: String,
    pub mode: SwarmMode,
    pub agents: Vec<AgentSpec>,
    /// Root permission set; every agent capability must be within this.
    pub permissions: SurfacePermissionSet,
    pub budget: SwarmBudget,
    pub usage: BudgetUsage,
    pub status: SwarmStatus,
    pub stop_reason: Option<StopReason>,
    pub receipts: Vec<AgentReceipt>,
    pub created_ms: u64,
    pub updated_ms: u64,
}

impl Swarm {
    /// Declare a swarm. Agents receive subset capabilities of `permissions`.
    pub fn declare(
        goal: impl Into<String>,
        mode: SwarmMode,
        permissions: SurfacePermissionSet,
        budget: SwarmBudget,
        now_ms: u64,
    ) -> Self {
        Self {
            id: SwarmId::new(),
            goal: goal.into(),
            mode,
            agents: Vec::new(),
            permissions,
            budget,
            usage: BudgetUsage::default(),
            status: SwarmStatus::Pending,
            stop_reason: None,
            receipts: Vec::new(),
            created_ms: now_ms,
            updated_ms: now_ms,
        }
    }

    /// Add an agent whose capability is derived as a subset of swarm permissions.
    pub fn add_agent(&mut self, mut spec: AgentSpec) -> Result<()> {
        if !matches!(self.status, SwarmStatus::Pending | SwarmStatus::Running) {
            return Err(YouError::InvalidState(format!(
                "swarm {} cannot add agents in status {:?}",
                self.id, self.status
            )));
        }
        // Derive capability: requested tools/connectors must be in swarm set.
        let cap = self.permissions.derive_capability_subset(
            spec.tools.iter().map(String::as_str),
            spec.connectors.iter().map(String::as_str),
        )?;
        // If tools/connectors empty, grant empty capability (still within set).
        if spec.tools.is_empty() && spec.connectors.is_empty() {
            spec.permissions = SurfaceCapability::default();
        } else {
            spec.permissions = cap;
        }
        if !spec.permissions.is_within(&self.permissions) {
            return Err(YouError::CapabilityMissing(
                "agent capability not within swarm permission set".into(),
            ));
        }

        // Agent-count ceiling: refuse before launch if already at max.
        if let Some(max) = self.budget.resources.max_agents {
            if (self.agents.len() as u32) >= max {
                return Err(YouError::BudgetExhausted(format!(
                    "max_agents={max} already reached"
                )));
            }
        }

        self.agents.push(spec);
        self.usage.agents_launched = self.agents.len() as u32;
        Ok(())
    }

    /// Convenience: build and add an agent with role + goal under swarm perms.
    pub fn spawn_role(
        &mut self,
        role: AgentRole,
        agent_goal: impl Into<String>,
        tools: impl IntoIterator<Item = impl Into<String>>,
        connectors: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<usize> {
        let tools: Vec<String> = tools.into_iter().map(Into::into).collect();
        let connectors: Vec<String> = connectors.into_iter().map(Into::into).collect();
        let cap = self
            .permissions
            .derive_capability_subset(tools.iter().map(String::as_str), connectors.iter().map(String::as_str))?;
        let spec = AgentSpec::builder(role, agent_goal)
            .tools(tools)
            .connectors(connectors)
            .permissions(cap)
            .budget(self.budget.resources.clone())
            .stop(self.budget.stop.clone())
            .context(serde_json::json!({
                "swarm_id": self.id.as_str(),
                "swarm_goal": self.goal,
                "mode": self.mode.as_str(),
            }))
            .build();
        self.add_agent(spec)?;
        Ok(self.agents.len() - 1)
    }

    pub fn may_run(&self) -> bool {
        matches!(self.status, SwarmStatus::Pending | SwarmStatus::Running)
    }

    /// Run all pending agents once via the fixture provider. Enforces budget
    /// after each agent; on exhaustion, halts and records why.
    pub fn run_round(&mut self, provider: &FixtureProvider, now_ms: u64) -> Result<Vec<AgentReceipt>> {
        if !self.may_run() {
            return Err(YouError::InvalidState(format!(
                "swarm {} is {:?}: {:?}",
                self.id, self.status, self.stop_reason
            )));
        }
        self.status = SwarmStatus::Running;
        self.updated_ms = now_ms;

        let mut round = Vec::new();
        // Snapshot agents to avoid borrow issues while mutating self.
        let agents: Vec<AgentSpec> = self.agents.clone();
        for spec in &agents {
            if let Some(reason) = self.check_stop() {
                self.halt(reason, now_ms);
                break;
            }
            let receipt = provider.run(spec);
            self.apply_receipt(&receipt);
            round.push(receipt.clone());
            self.receipts.push(receipt);

            if let Some(reason) = self.check_stop() {
                self.halt(reason, now_ms);
                break;
            }
        }

        if self.status == SwarmStatus::Running {
            // Round finished without budget halt.
            if self.agents.len() == self.receipts.len()
                || matches!(self.budget.stop, StopCondition::AfterSteps { .. })
            {
                // Leave Running if more rounds possible; mark Completed only
                // when stop says so or caller finishes explicitly.
            }
        }
        Ok(round)
    }

    /// Mark swarm completed when work is done under budget.
    pub fn complete(&mut self, now_ms: u64) -> Result<()> {
        if self.status == SwarmStatus::Halted {
            return Err(YouError::InvalidState(
                "cannot complete a halted swarm".into(),
            ));
        }
        self.status = SwarmStatus::Completed;
        self.stop_reason = Some(StopReason::Completed);
        self.updated_ms = now_ms;
        Ok(())
    }

    fn apply_receipt(&mut self, receipt: &AgentReceipt) {
        self.usage.tokens = self.usage.tokens.saturating_add(receipt.tokens_used);
        self.usage.steps = self.usage.steps.saturating_add(receipt.steps_used);
        self.usage.cpu_ms = self.usage.cpu_ms.saturating_add(receipt.cpu_ms);
        self.usage.wall_ms = self.usage.wall_ms.saturating_add(receipt.cpu_ms); // fixture: wall≈cpu
        if receipt.ram_mb > self.usage.ram_mb_peak {
            self.usage.ram_mb_peak = receipt.ram_mb;
        }
    }

    fn check_stop(&self) -> Option<StopReason> {
        if let Some(axis) = self.usage.exhausted_axis(&self.budget.resources) {
            return Some(StopReason::BudgetExhausted {
                axis: axis.as_str().to_string(),
            });
        }
        match &self.budget.stop {
            StopCondition::Never | StopCondition::BudgetOnly => None,
            StopCondition::AfterSteps { count } => {
                if self.usage.steps >= *count {
                    Some(StopReason::AfterSteps { count: *count })
                } else {
                    None
                }
            }
            StopCondition::AfterWallMs { ms } => {
                if self.usage.wall_ms >= *ms {
                    Some(StopReason::AfterWallMs { ms: *ms })
                } else {
                    None
                }
            }
            StopCondition::ConditionMet { .. } => None,
        }
    }

    fn halt(&mut self, reason: StopReason, now_ms: u64) {
        self.status = SwarmStatus::Halted;
        self.stop_reason = Some(reason);
        self.updated_ms = now_ms;
    }

    /// Signal an external stop condition.
    pub fn signal_condition(&mut self, name: &str, now_ms: u64) -> Result<()> {
        match &self.budget.stop {
            StopCondition::ConditionMet { name: expected } if expected == name => {
                self.halt(
                    StopReason::ConditionMet {
                        name: name.to_string(),
                    },
                    now_ms,
                );
                Ok(())
            }
            _ => Err(YouError::InvalidState(format!(
                "condition '{name}' is not this swarm's stop condition"
            ))),
        }
    }

    /// Inspectable declaration for contracts and UI.
    pub fn declaration(&self) -> Value {
        serde_json::json!({
            "id": self.id.as_str(),
            "goal": self.goal,
            "mode": self.mode.as_str(),
            "status": self.status,
            "stop_reason": self.stop_reason,
            "budget": self.budget,
            "usage": self.usage,
            "permissions": {
                "tools": self.permissions.tools().iter().cloned().collect::<Vec<_>>(),
                "connectors": self.permissions.connectors().iter().cloned().collect::<Vec<_>>(),
            },
            "agents": self.agents.iter().map(|a| serde_json::json!({
                "id": a.id.as_str(),
                "role": a.role.as_str(),
                "goal": a.goal,
                "tools": a.tools,
                "connectors": a.connectors,
                "model_profile": a.model_profile,
                "output_schema": a.output_schema.schema_id,
            })).collect::<Vec<_>>(),
            "receipt_count": self.receipts.len(),
        })
    }

    /// Shared goal + mode context capsule payload for agents.
    pub fn context_capsule_template(&self) -> Value {
        serde_json::json!({
            "swarm_id": self.id.as_str(),
            "goal": self.goal,
            "mode": self.mode.as_str(),
            "kind": "swarm_context",
        })
    }
}

/// Helper: default YOU swarm permission root (personal connectors read, no shell).
pub fn you_swarm_permissions() -> SurfacePermissionSet {
    SurfacePermissionSet::new(
        [
            "connector.read",
            "memory.read",
            "research.read",
            "object.read",
            "write.draft",
        ],
        ["gmail", "calendar", "personal_vault", "rss"],
    )
}

/// Helper: tight resource budget for tests that expect early halt.
pub fn test_budget(max_tokens: u64, max_steps: u32) -> SwarmBudget {
    SwarmBudget {
        resources: ResourceBudget {
            max_cpu_ms: Some(10_000),
            max_ram_mb: Some(256),
            max_tokens: Some(max_tokens),
            max_steps: Some(max_steps),
            max_wall_ms: Some(60_000),
            max_agents: Some(8),
        },
        stop: StopCondition::BudgetOnly,
    }
}
}

