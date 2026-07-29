use crate::approval::{ApprovalDecision, ApprovalHub};
use crate::commands::CommandRouter;
use crate::connectors::{register_backend_connectors, ConnectorRegistry, ConnectorStatus};
use crate::interrupt::InterruptHub;
use crate::memory::{
    MemoryDraft, MemoryLedger, MemoryRecord, MemoryRevalidation, MemoryScope, MemoryStatus,
    PrivacyClass, RevalidateTarget,
};
use crate::policy::{
    derive_policy_decision, tool_declared_effects, PolicyDecision, PolicyDecisionRecord,
};
use crate::process::{ProcessState, ProcessSupervisor, StartSpec};
use crate::replay::BackendReplayService;
use crate::rewind::{self, CheckpointCoverage, FileChange, ForkPoint, RewindTarget, StateRef};
use crate::security::SecurityServices;
use crate::initialize::{ClientCapabilities, ClientInfo, ConnectionRegistry, InitializeResponse};
use crate::live_thread::LiveThread;
use crate::services::{
    BackendCapabilities, BackendServices, Budget, CheckpointRecord, CheckpointStore,
    EnvironmentNode, EnvironmentSwitch, GoalOutcome, GoalRecord, GoalStatus, GoalStore, GoalVerdict,
    JobRecord, JobStatus, JobStore, RepoNode, SharedBackend, Trigger, TriggerEvent, TrustState,
    WorkspaceEdge, WorkspaceEdgeKind, WorkspaceGraph, WorkspaceStore,
};
use crate::supervisor::{RuntimeSupervisor, SupervisorConfig};
use crate::surfaces::SurfaceGraphService;
use crate::tools::{build_default_tool_dispatcher, build_default_tool_registry};
use crate::ui_bus::UiEventBus;
use hide_core::api::{Intent, IntentAck, UiEvent, UiEventKind};
use hide_core::event::{Event, NewEvent, ToolCallEvent, ToolResultEvent};
use hide_core::ids::{EventId, RunId, SessionId, StepId};
use hide_core::observability::{HealthCheck, HealthReport, HealthStatus};
use hide_core::runtime::{ModelRole, RuntimeSupervisorState};
use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry, ToolResult, ToolSpec, ToolStatus};
use hide_core::Result;
use hide_fleet::manager::KernelRunLauncher;
use hide_fleet::{
    AgentJob, ConcurrencyClass, FleetConfig, FleetGovernor, FleetManager, OsResourceProbe,
    PriorityClass,
};
use hide_kernel::govern::{Autonomy, Interrupt};
use hide_kernel::machine::state::{AgentState, ApprovalRequest, Phase};
use hide_kernel::session::SessionProjection;
use hide_kernel::{AgentKernel, Grounding};
// Bible Book IX sec 28-29 / sec 78.1 #6: the deterministic verification plane.
// The colliding names (`Verdict`, `VerificationInput`, `Oracle`) are qualified
// as `hide_kernel::verify_plane::*` at their (few) use sites so the function-local
// `hide_kernel::verify::oracle::*` imports in the goal path and the tests keep
// their meaning; only the non-colliding types are imported here.
use hide_kernel::verify_plane::{
    Finding, GateDecision, ReviewRole, ReviewRoleProfile, SourceFile, StaticAnalysisOracle,
    TieredVerdict, VerificationReceipt, VerificationTier,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use super::*;

impl BackendHost {

    /// REPLAY from a checkpoint: re-apply the whole recorded history from the
    /// checkpoint forward onto a fresh, independent lineage seeded at the
    /// checkpoint (behind a [`ForkPoint`] marker). The post-boundary source events
    /// are the replayed set (the child's own records). Unlike a rewind, replay
    /// drops nothing. Model-free.
    pub async fn checkpoint_replay(&self, checkpoint_id: &str) -> Result<ReplayOutcome> {
        let record = self.load_verified_checkpoint(checkpoint_id)?;
        let source = record.session_id.clone();
        let at_seq = record.at_seq;
        let events = self
            .services
            .event_log
            .scan(Some(source.clone()), None, None)
            .await?;
        let child_events: Vec<&Event> = events.iter().collect();
        let replayed_events: Vec<EventId> = events
            .iter()
            .filter(|e| e.seq > at_seq)
            .map(|e| e.id.clone())
            .collect();
        let inherited = rewind::inherited_len(&events, at_seq);
        let (fork_point, marker) = self.fork_marker(&source, inherited, at_seq);
        let (child, projection) = self
            .replay
            .seed_child_session(Some(marker), &child_events)
            .await?;
        let ancestry = crate::services::SessionRecord::fork(
            child.clone(),
            source.clone(),
            at_seq,
            record.at_event.clone(),
        );
        self.services
            .sessions
            .record_session(&self.services.key_value_store, &ancestry);
        self.publish_checkpoint_child(
            "checkpoint_replayed",
            &child,
            &record,
            json!({ "replayed": replayed_events.len(), "fork_point": fork_point }),
        );
        Ok(ReplayOutcome {
            session_id: child,
            fork_point,
            replayed_events,
            projection,
            ancestry,
        })
    }

    /// FORK from a checkpoint into an ephemeral branch: a new lineage seeded ONLY
    /// with the checkpoint's inherited prefix (behind a [`ForkPoint`] marker), to
    /// explore an alternative from the boundary with no post-boundary carry-over.
    /// Recorded as an [`SessionRelationship::EphemeralFork`](crate::services::SessionRelationship)
    /// so a client can prune it without ceremony. Model-free.
    pub async fn checkpoint_fork(&self, checkpoint_id: &str) -> Result<ForkOutcome> {
        let record = self.load_verified_checkpoint(checkpoint_id)?;
        let source = record.session_id.clone();
        let at_seq = record.at_seq;
        let events = self
            .services
            .event_log
            .scan(Some(source.clone()), None, None)
            .await?;
        let child_events: Vec<&Event> = events.iter().filter(|e| e.seq <= at_seq).collect();
        let inherited = child_events.len();
        let (fork_point, marker) = self.fork_marker(&source, inherited, at_seq);
        let (child, projection) = self
            .replay
            .seed_child_session(Some(marker), &child_events)
            .await?;
        let ancestry = crate::services::SessionRecord::ephemeral_fork(
            child.clone(),
            source.clone(),
            at_seq,
            record.at_event.clone(),
        );
        self.services
            .sessions
            .record_session(&self.services.key_value_store, &ancestry);
        self.publish_checkpoint_child(
            "checkpoint_forked",
            &child,
            &record,
            json!({ "fork_point": fork_point }),
        );
        Ok(ForkOutcome {
            session_id: child,
            fork_point,
            projection,
            ancestry,
        })
    }

    /// COMPARE a session's current code state against a checkpoint's boundary code
    /// state (current-versus-checkpoint): the file-level added/removed/modified
    /// changes. Model-free.
    pub async fn compare_to_checkpoint(
        &self,
        checkpoint_id: &str,
        session: &SessionId,
    ) -> Result<CodeComparison> {
        let record = CheckpointStore::get(&self.services.key_value_store, checkpoint_id)
            .ok_or_else(|| {
                hide_core::error::HideError::NotFound(format!("unknown checkpoint {checkpoint_id}"))
            })?;
        let base = self
            .code_state_of(&record.session_id, Some(record.at_seq))
            .await?;
        let head = self.code_state_of(session, None).await?;
        Ok(CodeComparison {
            base: format!("checkpoint:{}", record.checkpoint_id),
            head: format!("session:{}", session.as_str()),
            files: rewind::diff_code_states(&base, &head),
        })
    }

    /// COMPARE two sessions' current code states (compare branches). Model-free.
    pub async fn compare_session_code(
        &self,
        a: &SessionId,
        b: &SessionId,
    ) -> Result<CodeComparison> {
        let base = self.code_state_of(a, None).await?;
        let head = self.code_state_of(b, None).await?;
        Ok(CodeComparison {
            base: format!("session:{}", a.as_str()),
            head: format!("session:{}", b.as_str()),
            files: rewind::diff_code_states(&base, &head),
        })
    }

    /// INSPECT a checkpoint's integrity + coverage (consolidation Trace E part d):
    /// whether the sealed digest verifies, whether the coverage recomputed from the
    /// current source log still matches (drift detection), and which verification
    /// receipts a code rewind from here would invalidate. Model-free.
    pub async fn checkpoint_inspect(&self, checkpoint_id: &str) -> Result<CheckpointInspection> {
        let record = CheckpointStore::get(&self.services.key_value_store, checkpoint_id)
            .ok_or_else(|| {
                hide_core::error::HideError::NotFound(format!("unknown checkpoint {checkpoint_id}"))
            })?;
        let integrity_ok = record.verify_integrity();
        let current = self.compute_coverage(&record.session_id, record.at_seq).await?;
        let drift = coverage_drift(&record.coverage, &current);

        let events = self
            .services
            .event_log
            .scan(Some(record.session_id.clone()), None, None)
            .await?;
        let base = rewind::code_state(&events, Some(record.at_seq));
        let head = rewind::code_state(&events, None);
        let reverted_files = rewind::changed_files(&base, &head);
        let receipts = rewind::receipt_scopes(&events, record.at_seq);
        let invalidated_receipts = rewind::invalidated_receipts(&reverted_files, &receipts);

        Ok(CheckpointInspection {
            checkpoint_id: record.checkpoint_id.clone(),
            integrity_ok,
            coverage_current: drift.is_empty(),
            drift,
            reverted_files,
            invalidated_receipts,
            coverage: record.coverage.clone(),
        })
    }

    /// Publish a checkpoint-child lifecycle UiEvent (rewound / replayed / forked)
    /// under the child session, carrying the source checkpoint + operation detail.
    pub(crate) fn publish_checkpoint_child(
        &self,
        kind: &str,
        child: &SessionId,
        checkpoint: &CheckpointRecord,
        detail: Value,
    ) {
        self.ui_bus.publish(UiEvent {
            seq: checkpoint.at_seq,
            session_id: Some(child.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": kind,
                "session_id": child.as_str(),
                "checkpoint": serde_json::to_value(checkpoint).unwrap_or_else(|_| json!({})),
                "detail": detail,
            })),
        });
    }

    // --- Durable background JOBS + triggers (bible sec 73-75, sec 78.1 #17) ---

    /// Create a durable BACKGROUND JOB: a goal-bound record that survives a
    /// restart. The `job` is written to the KV `jobs` namespace keyed by its
    /// `job_id`, and a `job.created` event is appended to the session's durable
    /// event log so the record is BOUND to that log (auditable + recoverable). A
    /// `job_created` UiEvent is surfaced under the session. Build the record via
    /// [`JobRecord::pending`] (plus its builders) so the id + timestamps are set.
    ///
    /// This persists the durable record only. The ACTUAL agent execution of the
    /// job when a trigger fires is DEFERRED_MODEL_REQUIRED: this method never runs
    /// a model or spawns an agent.
    pub async fn job_create(&self, job: JobRecord) -> Result<JobRecord> {
        JobStore::put(&self.services.key_value_store, &job)?;
        self.services
            .event_log
            .append(NewEvent::system(
                job.session_id.clone(),
                "job.created",
                serde_json::to_value(&job).unwrap_or(Value::Null),
            ))
            .await?;
        self.publish_job(&job, "job_created");
        Ok(job)
    }

    /// The durable job with `job_id`, if any.
    pub fn job_get(&self, job_id: &str) -> Option<JobRecord> {
        JobStore::get(&self.services.key_value_store, job_id)
    }

    /// Every durable job, ordered deterministically (created_ms then job_id).
    pub fn job_list(&self) -> Vec<JobRecord> {
        JobStore::list_all(&self.services.key_value_store)
    }

    /// DETERMINISTICALLY evaluate whether an incoming `event` matches ANY trigger
    /// on `job` (bible sec 75) -- the wake predicate. NO model; a pure function of
    /// the job's triggers and the event's kind + payload (a `FileChange` glob is
    /// matched against the event path, a `Manual` trigger fires only on a `Manual`
    /// event, etc.).
    ///
    /// A `true` here means the job SHOULD be dispatched; the actual dispatch /
    /// execution is DEFERRED_MODEL_REQUIRED and is not performed by this call.
    pub fn job_evaluate_triggers(&self, job: &JobRecord, event: &TriggerEvent) -> bool {
        job.matches_event(event)
    }

    /// Durably transition a job's `status` (bible sec 73), stamping `updated_ms`
    /// and recording an optional `last_error`. The updated record is written back
    /// to KV, a `job.status` event is appended to the session log, and a
    /// `job_status` UiEvent is surfaced. Returns the updated record, or `None` when
    /// no such job exists.
    pub async fn job_update_status(
        &self,
        job_id: &str,
        status: JobStatus,
        last_error: Option<String>,
    ) -> Result<Option<JobRecord>> {
        let kv = &self.services.key_value_store;
        match JobStore::get(kv, job_id) {
            Some(mut job) => {
                job.status = status;
                job.updated_ms = hide_core::ids::now_ms();
                if last_error.is_some() {
                    job.last_error = last_error;
                }
                JobStore::put(kv, &job)?;
                self.services
                    .event_log
                    .append(NewEvent::system(
                        job.session_id.clone(),
                        "job.status",
                        serde_json::to_value(&job).unwrap_or(Value::Null),
                    ))
                    .await?;
                self.publish_job(&job, "job_status");
                Ok(Some(job))
            }
            None => Ok(None),
        }
    }

    /// Cancel a job: flip its status to `Cancelled` durably (a terminal state
    /// excluded from `jobs_recover`), append a `job.cancelled` event, and surface a
    /// `job_cancelled` UiEvent. Returns the cancelled record, or `None` when no
    /// such job exists.
    pub async fn job_cancel(&self, job_id: &str) -> Result<Option<JobRecord>> {
        let kv = &self.services.key_value_store;
        match JobStore::get(kv, job_id) {
            Some(mut job) => {
                job.status = JobStatus::Cancelled;
                job.updated_ms = hide_core::ids::now_ms();
                JobStore::put(kv, &job)?;
                self.services
                    .event_log
                    .append(NewEvent::system(
                        job.session_id.clone(),
                        "job.cancelled",
                        serde_json::to_value(&job).unwrap_or(Value::Null),
                    ))
                    .await?;
                self.publish_job(&job, "job_cancelled");
                Ok(Some(job))
            }
            None => Ok(None),
        }
    }

    /// Rebuild the ACTIVE background-job set from the durable store on startup
    /// (bible sec 73): every job whose status is Pending / Running / Blocked,
    /// ordered deterministically. Terminal jobs (Done / Cancelled / Failed) are
    /// excluded. A fresh [`BackendHost`] over the same workspace recovers exactly
    /// this set, so scheduled / triggered jobs SURVIVE A RESTART. Re-dispatching a
    /// recovered job when its trigger next fires is DEFERRED_MODEL_REQUIRED.
    pub fn jobs_recover(&self) -> Vec<JobRecord> {
        JobStore::recover(&self.services.key_value_store)
    }

    /// Publish a job-lifecycle UiEvent (`job_created` / `job_status` /
    /// `job_cancelled`) carrying the record, under the job's session.
    pub(crate) fn publish_job(&self, record: &JobRecord, kind: &str) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(record.session_id.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": kind,
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    // --- Stage 4: durable-thread lifecycle + Initialize + background promotion ---

    /// Open a durable-thread writer (Stage 4 four-verb lifecycle) over a session's
    /// event log. Appended items are lazy until an explicit `flush` / `persist` /
    /// `shutdown`; `discard` drops them without a durable write. Wrap it in a
    /// [`crate::live_thread::LiveThreadInitGuard`] to make a failed session init
    /// discard its partial event stream.
    pub fn open_live_thread(&self, session: SessionId) -> LiveThread {
        LiveThread::open(session, self.services.event_log.clone())
    }

    /// Handle a client Initialize (Stage 4 capability negotiation, Codex mechanism
    /// 5). Records the connection's negotiated `capabilities` (the experimental-api
    /// gate + the opt-out notification method set) keyed by `connection_id`, and
    /// returns the server-info reply. The stored capabilities are consulted in the
    /// notification emit path ([`Self::notification_for_connection`]). The
    /// `ClientInfo` is accepted per the handshake but not retained (only the
    /// negotiation levers drive server behavior). Model-free.
    pub fn initialize(
        &self,
        connection_id: impl Into<String>,
        _client: ClientInfo,
        capabilities: ClientCapabilities,
    ) -> InitializeResponse {
        self.connections.initialize(connection_id, capabilities);
        InitializeResponse {
            user_agent: format!("hide-backend/{}", env!("CARGO_PKG_VERSION")),
            workspace_root: self.services.config.workspace_root.display().to_string(),
            platform_family: std::env::consts::FAMILY.to_string(),
            platform_os: std::env::consts::OS.to_string(),
        }
    }

    /// The per-connection capability registry (Stage 4 Initialize handshake). The
    /// notification emit path consults it to suppress opted-out methods.
    pub fn connections(&self) -> &ConnectionRegistry {
        &self.connections
    }

    /// Promote a LIVE interactive run to a durable BACKGROUND JOB (Stage 4
    /// background promotion) WITHOUT restarting it: the still-running run keeps its
    /// `run_id` and its tokio task, so it survives a client disconnect. A durable
    /// [`JobRecord`] bound to that run id is created (status `Running`, a Manual
    /// wake trigger), so a fresh host recovers it and a reconnecting client can
    /// find, inspect, steer, pause, stop, fork, and resume-in-foreground the SAME
    /// run. Also appends a `run.promoted` event tying the run to the job on the
    /// session log. Reuses `job_create` (never a second store); model-free.
    pub async fn promote_run_to_background(
        &self,
        run_id: RunId,
        session: SessionId,
        goal_id: Option<String>,
        budget: Budget,
    ) -> Result<JobRecord> {
        let mut job =
            JobRecord::pending(session.clone(), vec![Trigger::Manual], budget).with_run(run_id.as_str());
        // The run is already executing: the promoted job is Running, not Pending.
        job.status = JobStatus::Running;
        if let Some(goal) = goal_id {
            job = job.with_goal(goal);
        }
        // Reuse the durable job path (writes the record + a `job.created` event +
        // publishes `job_created`), then tie the run to the job on the log.
        let job = self.job_create(job).await?;
        self.services
            .event_log
            .append(
                NewEvent::system(
                    session,
                    "run.promoted",
                    json!({ "run_id": run_id.as_str(), "job_id": job.job_id }),
                )
                .with_run(run_id.clone()),
            )
            .await?;
        self.publish_job(&job, "job_promoted");
        Ok(job)
    }

    /// The durable background job bound to a live `run_id`, if the run was promoted
    /// (Stage 4). Deterministic scan of the durable job store; survives a restart
    /// because the binding lives in the persisted [`JobRecord::run_id`].
    pub fn background_job_for_run(&self, run_id: &RunId) -> Option<JobRecord> {
        self.job_list()
            .into_iter()
            .find(|job| job.run_id.as_deref() == Some(run_id.as_str()))
    }

    /// Inspect the ARTIFACTS a background job accumulated (Stage 4 inspect): the
    /// durable job record, the run's own events replayed from the session log
    /// (filtered to the promoted `run_id`), and the checkpoints pinned on its
    /// session. A read-only, model-free projection a reconnecting client uses to
    /// see what the background run produced while it was detached. Errors if
    /// `job_id` is unknown.
    pub async fn background_job_artifacts(&self, job_id: &str) -> Result<Value> {
        let job = self.job_get(job_id).ok_or_else(|| {
            hide_core::error::HideError::Message(format!(
                "background_job_artifacts: no such job '{job_id}'"
            ))
        })?;
        let run_id = job.run_id.clone();
        let run_events: Vec<Event> = self
            .services
            .event_log
            .scan(Some(job.session_id.clone()), None, None)
            .await?
            .into_iter()
            .filter(|e| {
                run_id
                    .as_deref()
                    .map(|r| e.run_id.as_ref().map(|er| er.as_str()) == Some(r))
                    .unwrap_or(false)
            })
            .collect();
        let checkpoints = self.checkpoint_list(&job.session_id);
        Ok(json!({
            "job": job,
            "run_events": run_events,
            "checkpoints": checkpoints,
        }))
    }

    /// Resume a promoted background job IN THE FOREGROUND (Stage 4): the
    /// reconnecting client reattaches to the still-running run. Clears any buffered
    /// pause on the run (the run continues, mirroring `ResumeRun`), flips the job
    /// status back to `Running`, appends a durable `run.resumed_foreground` event,
    /// republishes the session projection so the reattached client re-renders the
    /// transcript it missed, and returns `(job, projection)`. Errors if the job is
    /// unknown or was never bound to a run. Model-free.
    pub async fn resume_background_job_in_foreground(
        &self,
        job_id: &str,
    ) -> Result<(JobRecord, SessionProjection)> {
        let job = self.job_get(job_id).ok_or_else(|| {
            hide_core::error::HideError::Message(format!(
                "resume_background_job_in_foreground: no such job '{job_id}'"
            ))
        })?;
        let run_id = job.run_id.clone().ok_or_else(|| {
            hide_core::error::HideError::Message(format!(
                "resume_background_job_in_foreground: job '{job_id}' is not bound to a run"
            ))
        })?;
        // Continue the run: clear any buffered pause (same as ResumeRun).
        self.interrupts.clear(&RunId::from(run_id.as_str()));
        // The job is foregrounded and active again.
        let job = self
            .job_update_status(job_id, JobStatus::Running, None)
            .await?
            .unwrap_or(job);
        // Durable foreground-resume marker on the run's session log.
        self.services
            .event_log
            .append(
                NewEvent::system(
                    job.session_id.clone(),
                    "run.resumed_foreground",
                    json!({ "run_id": run_id, "job_id": job.job_id }),
                )
                .with_run(RunId::from(run_id.as_str())),
            )
            .await?;
        // Replay: rebuild + return the projection so the reattached client
        // re-renders the transcript it missed while detached.
        let projection = self
            .rebuild_session_projection(job.session_id.clone())
            .await?;
        self.publish_job(&job, "job_resumed_foreground");
        Ok((job, projection))
    }

    /// Dispatch a Stage 4 background-promotion custom intent to the corresponding
    /// host method (mirrors [`Self::handle_memory_workspace_env_intent`]). Payload
    /// shapes:
    ///
    /// * `promote_run`           -> `{ run_id, session_id, goal_id?, budget? }`
    /// * `resume_run_foreground` -> `{ job_id }`
    pub(crate) async fn handle_background_intent(&self, name: &str, payload: &Value) -> Result<()> {
        let missing = |field: &str| {
            hide_core::error::HideError::Message(format!("{name}: missing '{field}'"))
        };
        match name {
            "promote_run" => {
                let run_id = payload
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("run_id"))?;
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("session_id"))?;
                let goal_id = payload
                    .get("goal_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let budget = payload
                    .get("budget")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                self.promote_run_to_background(
                    RunId::from(run_id),
                    SessionId::from(session),
                    goal_id,
                    budget,
                )
                .await?;
            }
            "resume_run_foreground" => {
                let job_id = payload
                    .get("job_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("job_id"))?;
                self.resume_background_job_in_foreground(job_id).await?;
            }
            _ => {}
        }
        Ok(())
    }

    // --- Outcome-governed durable MEMORY + revalidation (bible sec 21-22, sec 78.1 #16) ---

    /// Add a durable, provenance-carrying MEMORY record (bible sec 21-22). The
    /// record is minted `Active` at the neutral outcome score from the `draft`
    /// (scope + claim + source + author + citations + privacy + optional expiry)
    /// and written to the KV `memory` namespace keyed by its minted id, so it
    /// survives a workspace reopen. Returns the stored record (with its id).
    pub fn memory_add(&self, draft: MemoryDraft) -> Result<MemoryRecord> {
        let record = MemoryRecord::from_draft(draft);
        MemoryLedger::put(&self.services.key_value_store, &record)?;
        // Mirror explicit durable memory into the six-class stores by scope:
        // User → user class (sole UserWriteCap mint); Repo/Session → semantic_project.
        // Never verification (verifier path only).
        let session_id = match &record.scope {
            MemoryScope::Session(id) => Some(id.as_str()),
            _ => None,
        };
        crate::classed_writers::mirror_memory_ledger_to_classes(
            &self.services.classed_memory,
            record.scope.kind(),
            &record.claim,
            &record.source,
            &record.author,
            &record.citations,
            session_id,
        );
        Ok(record)
    }

    /// The durable memory record with `memory_id`, if any.
    pub fn memory_get(&self, memory_id: &str) -> Option<MemoryRecord> {
        MemoryLedger::get(&self.services.key_value_store, memory_id)
    }

    /// Every memory record BOUND to `scope`, ordered deterministically (created_ms
    /// then id). Scope equality is by value, so a `Repo` list never returns a
    /// `Session` or `User` record. Returns every status (Active / Quarantined /
    /// Superseded); use [`Self::memory_context`] for the context-eligible subset.
    pub fn memory_list(&self, scope: &MemoryScope) -> Vec<MemoryRecord> {
        MemoryLedger::list_scope(&self.services.key_value_store, scope)
    }

    /// The records in `scope` that are eligible to ENTER CONTEXT: `Active` and not
    /// expired. This is the only set the context compiler should draw from (bible
    /// sec 21-22): a quarantined or superseded or expired claim never re-enters.
    pub fn memory_context(&self, scope: &MemoryScope) -> Vec<MemoryRecord> {
        let now = hide_core::ids::now_ms();
        MemoryLedger::list_scope(&self.services.key_value_store, scope)
            .into_iter()
            .filter(|record| record.is_eligible(now))
            .collect()
    }

    /// SUPERSEDE a record with a replacement (bible sec 21-22) WITHOUT erasing
    /// history: the old record is marked `Superseded` and linked to the new one
    /// (`superseded_by`), the new record links back (`supersedes`), and BOTH are
    /// persisted. The old record stays in the ledger (queryable, auditable); it is
    /// simply no longer context-eligible. Returns `(old_superseded, new_active)`.
    /// `NotFound` when `old_id` is unknown.
    pub fn memory_supersede(
        &self,
        old_id: &str,
        replacement: MemoryDraft,
    ) -> Result<(MemoryRecord, MemoryRecord)> {
        let kv = &self.services.key_value_store;
        let mut old = MemoryLedger::get(kv, old_id).ok_or_else(|| {
            hide_core::error::HideError::NotFound(format!("unknown memory record {old_id}"))
        })?;
        let mut new = MemoryRecord::from_draft(replacement);
        new.supersedes = Some(old.memory_id.clone());
        old.status = MemoryStatus::Superseded;
        old.superseded_by = Some(new.memory_id.clone());
        // Write the new record first, then the retired old one, so a crash between
        // the two never leaves the old record pointing at a nonexistent successor.
        MemoryLedger::put(kv, &new)?;
        MemoryLedger::put(kv, &old)?;
        Ok((old, new))
    }

    /// Record an OUTCOME of exercising a memory claim (bible sec 21-22
    /// governance). A success raises the record's `outcome_score` and `use_count`;
    /// a failure lowers the score and, once it falls below the quarantine floor,
    /// flips the record to `Quarantined` so it stops entering context. The updated
    /// record is persisted and returned. `NotFound` when `memory_id` is unknown.
    pub fn memory_record_outcome(&self, memory_id: &str, success: bool) -> Result<MemoryRecord> {
        let kv = &self.services.key_value_store;
        let mut record = MemoryLedger::get(kv, memory_id).ok_or_else(|| {
            hide_core::error::HideError::NotFound(format!("unknown memory record {memory_id}"))
        })?;
        record.record_outcome(success);
        MemoryLedger::put(kv, &record)?;
        Ok(record)
    }

    /// REVALIDATE a memory record (or a whole scope) against the CURRENT repo on
    /// disk (bible sec 21-22). For each `Active` target record, every citation is
    /// checked with [`crate::memory::resolve_citation`]: a cited `path` must exist,
    /// and a `path#symbol` file must exist AND contain the symbol via a lexical
    /// scan. A record with an unresolved citation is QUARANTINED (durably) with a
    /// reason; a record whose citations all still resolve keeps its status and has
    /// its `last_validated_ms` bumped. Non-`Active` records are reported but not
    /// mutated.
    ///
    /// This is deterministic and MODEL-FREE. SEMANTIC revalidation -- judging
    /// whether a claim is still true in spirit even when its citations resolve --
    /// is `DEFERRED_MODEL_REQUIRED`: no model is ever loaded or called here.
    ///
    /// Returns one [`MemoryRevalidation`] verdict per record considered. A
    /// `RevalidateTarget::Record` with an unknown id errors `NotFound`; an empty
    /// scope returns an empty vec.
    pub fn memory_revalidate(
        &self,
        target: RevalidateTarget,
        repo_root: &std::path::Path,
    ) -> Result<Vec<MemoryRevalidation>> {
        let kv = &self.services.key_value_store;
        let records: Vec<MemoryRecord> = match target {
            RevalidateTarget::Record(memory_id) => {
                let record = MemoryLedger::get(kv, &memory_id).ok_or_else(|| {
                    hide_core::error::HideError::NotFound(format!(
                        "unknown memory record {memory_id}"
                    ))
                })?;
                vec![record]
            }
            RevalidateTarget::Scope(scope) => MemoryLedger::list_scope(kv, &scope),
        };

        let now = hide_core::ids::now_ms();
        let mut out = Vec::with_capacity(records.len());
        for mut record in records {
            let unresolved: Vec<String> = record
                .citations
                .iter()
                .map(|citation| crate::memory::resolve_citation(repo_root, citation))
                .filter(|resolution| !resolution.resolved)
                .map(|resolution| resolution.citation)
                .collect();
            let resolved = unresolved.is_empty();
            // Only an Active record transitions; a Quarantined/Superseded record is
            // reported but never re-mutated by a revalidation pass.
            let was_active = record.status == MemoryStatus::Active;
            let mut quarantined = false;
            if was_active && !resolved {
                record.status = MemoryStatus::Quarantined;
                record.last_validated_ms = now;
                quarantined = true;
                MemoryLedger::put(kv, &record)?;
            } else if was_active && resolved {
                record.last_validated_ms = now;
                MemoryLedger::put(kv, &record)?;
            }
            let reason = if resolved {
                "all citations resolve against the repo on disk".to_string()
            } else {
                format!(
                    "citation(s) no longer resolve: {}",
                    unresolved.join(", ")
                )
            };
            out.push(MemoryRevalidation {
                memory_id: record.memory_id,
                status: record.status,
                resolved,
                unresolved,
                reason,
                quarantined,
            });
        }
        Ok(out)
    }

    /// Side-chat lifecycle (bible sec 32-33, sec 78.1 #9) -- CREATE. Fork a
    /// [`SessionRelationship::SideChat`] from `parent_session` at `at_event`
    /// (`None` = the parent's current tail), recorded READ-ONLY by default with
    /// ancestry preserved. When `inherit` is true the side chat sees the
    /// pre-boundary history; when false it starts empty (ancestry only). The
    /// parent is UNTOUCHED (independent lineage). Surfaces a `side_chat_created`
    /// UiEvent under the new session id.
    pub async fn create_side_chat(
        &self,
        parent_session: SessionId,
        at_event: Option<&EventId>,
        inherit: bool,
    ) -> Result<(SessionId, crate::services::SessionRecord, SessionProjection)> {
        let (new_session, record, projection) = branch_and_record(
            &self.replay,
            &self.services.sessions,
            &self.services.key_value_store,
            parent_session,
            at_event.cloned(),
            crate::services::SessionRelationship::SideChat,
            true, // read-only by default
            inherit,
        )
        .await?;
        self.publish_side_chat_created(&new_session, &record);
        Ok((new_session, record, projection))
    }

    /// Side-chat lifecycle -- MERGE. Append a durable `session.merge_summary`
    /// event ONTO THE PARENT session, carrying the `side_chat` id + the typed
    /// `summary_text`. The parent's transcript gains a cited summary (a
    /// parent-scoped [`Self::search_transcript`] surfaces it); the side chat is
    /// NOT destroyed -- its own event lineage is left entirely intact. Surfaces a
    /// `side_chat_merged` UiEvent under the parent. Returns the appended event.
    ///
    /// Discarding a side chat is simply the absence of this call: with no merge,
    /// nothing is appended and the parent stays exactly as it was.
    pub async fn merge_side_chat_summary(
        &self,
        side_chat: SessionId,
        parent: SessionId,
        summary_text: impl Into<String>,
    ) -> Result<Event> {
        self.merge_side_chat_result(side_chat, parent, SideChatResult::summary_only(summary_text))
            .await
    }

    /// Side-chat lifecycle -- MERGE a CONCISE TYPED result (bible sec 32-33, sec
    /// 78.1 #9). Append a durable `session.merge_summary` event ONTO THE PARENT
    /// carrying the side chat id + the typed [`SideChatResult`] (summary + cited
    /// `evidence` links + `kind`). The parent gains this BOUNDED result, NEVER the
    /// child's full transcript; the side chat's own lineage is left entirely
    /// intact. `summary` stays at the payload top level so a parent-scoped
    /// [`Self::search_transcript`] surfaces the cited summary. Surfaces a
    /// `side_chat_merged` UiEvent under the parent. Returns the appended event.
    pub async fn merge_side_chat_result(
        &self,
        side_chat: SessionId,
        parent: SessionId,
        result: SideChatResult,
    ) -> Result<Event> {
        let event = self
            .services
            .event_log
            .append(NewEvent::system(
                parent.clone(),
                "session.merge_summary",
                result.merge_event_payload(&side_chat),
            ))
            .await?;
        self.publish_side_chat_merged(&parent, &side_chat, &result, event.seq);
        Ok(event)
    }

    /// Publish a `side_chat_created` UiEvent carrying the new thread's record,
    /// under the new session id (mirrors [`Self::publish_session_forked`]).
    pub(crate) fn publish_side_chat_created(
        &self,
        new_session: &SessionId,
        record: &crate::services::SessionRecord,
    ) {
        self.ui_bus.publish(UiEvent {
            seq: record.forked_at.unwrap_or(0),
            session_id: Some(new_session.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "side_chat_created",
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    /// Publish a `side_chat_merged` UiEvent under the PARENT (the merge lands on
    /// the parent, not the side chat), carrying both ids + the merged summary.
    pub(crate) fn publish_side_chat_merged(
        &self,
        parent: &SessionId,
        side_chat: &SessionId,
        result: &SideChatResult,
        seq: u64,
    ) {
        self.ui_bus.publish(UiEvent {
            seq,
            session_id: Some(parent.clone()),
            kind: UiEventKind::Custom(result.merged_ui_payload(parent, side_chat)),
        });
    }

    /// Perform a `create_side_chat` custom intent: spawn the side-chat creation so
    /// the intent ack returns immediately (mirrors [`Self::spawn_fork_session`]).
    /// A failure (e.g. an unknown boundary event) surfaces as an Error UiEvent.
    pub(crate) fn spawn_create_side_chat(
        &self,
        parent: SessionId,
        at_event: Option<EventId>,
        inherit: bool,
    ) {
        let replay = self.replay.clone();
        let sessions = self.services.sessions.clone();
        let kv = self.services.key_value_store.clone();
        let bus = Arc::clone(&self.ui_bus);
        tokio::spawn(async move {
            match branch_and_record(
                &replay,
                &sessions,
                &kv,
                parent.clone(),
                at_event,
                crate::services::SessionRelationship::SideChat,
                true,
                inherit,
            )
            .await
            {
                Ok((new_session, record, _)) => {
                    bus.publish(UiEvent {
                        seq: record.forked_at.unwrap_or(0),
                        session_id: Some(new_session),
                        kind: UiEventKind::Custom(json!({
                            "kind": "side_chat_created",
                            "record": serde_json::to_value(&record).unwrap_or_else(|_| json!({})),
                        })),
                    });
                }
                Err(err) => {
                    bus.publish(UiEvent {
                        seq: 0,
                        session_id: Some(parent),
                        kind: UiEventKind::Error {
                            code: "create_side_chat".to_string(),
                            message: err.to_string(),
                        },
                    });
                }
            }
        });
    }

    /// Perform a `merge_side_chat` custom intent: append the merge summary onto
    /// the parent (spawned; the ack returns immediately). Surfacing is done by
    /// [`Self::merge_side_chat_summary`]; a failure surfaces as an Error UiEvent.
    pub(crate) fn spawn_merge_side_chat(&self, side_chat: SessionId, parent: SessionId, summary: String) {
        let event_log = self.services.event_log.clone();
        let bus = Arc::clone(&self.ui_bus);
        let result = SideChatResult::summary_only(summary);
        tokio::spawn(async move {
            let appended = event_log
                .append(NewEvent::system(
                    parent.clone(),
                    "session.merge_summary",
                    result.merge_event_payload(&side_chat),
                ))
                .await;
            match appended {
                Ok(event) => {
                    bus.publish(UiEvent {
                        seq: event.seq,
                        session_id: Some(parent.clone()),
                        kind: UiEventKind::Custom(result.merged_ui_payload(&parent, &side_chat)),
                    });
                }
                Err(err) => {
                    bus.publish(UiEvent {
                        seq: 0,
                        session_id: Some(parent),
                        kind: UiEventKind::Error {
                            code: "merge_side_chat".to_string(),
                            message: err.to_string(),
                        },
                    });
                }
            }
        });
    }

    /// Publish a `session_forked` UiEvent carrying the new thread's record, under
    /// the new session id (so the FE, which adopts a session off any event's
    /// `session_id`, switches to the fork).
    pub(crate) fn publish_session_forked(&self, new_session: &SessionId, record: &crate::services::SessionRecord) {
        self.ui_bus.publish(UiEvent {
            seq: record.forked_at.unwrap_or(0),
            session_id: Some(new_session.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "session_forked",
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    /// Perform a `ForkSession` intent: fork the source at the event boundary,
    /// record ancestry, and surface the new thread. Spawned so the intent ack
    /// returns immediately (mirrors [`Self::spawn_open_session`]); a failure (e.g.
    /// an unknown boundary event) surfaces as an Error UiEvent, never a panic.
    pub(crate) fn spawn_fork_session(&self, from: SessionId, at_event: hide_core::ids::EventId) {
        let replay = self.replay.clone();
        let sessions = self.services.sessions.clone();
        let kv = self.services.key_value_store.clone();
        let bus = Arc::clone(&self.ui_bus);
        tokio::spawn(async move {
            match fork_and_record(&replay, &sessions, &kv, from.clone(), Some(at_event)).await {
                Ok((new_session, record, _)) => {
                    bus.publish(UiEvent {
                        seq: record.forked_at.unwrap_or(0),
                        session_id: Some(new_session),
                        kind: UiEventKind::Custom(json!({
                            "kind": "session_forked",
                            "record": serde_json::to_value(&record).unwrap_or_else(|_| json!({})),
                        })),
                    });
                }
                Err(err) => {
                    bus.publish(UiEvent {
                        seq: 0,
                        session_id: Some(from),
                        kind: UiEventKind::Error {
                            code: "fork_session".to_string(),
                            message: err.to_string(),
                        },
                    });
                }
            }
        });
    }

    pub async fn status(&self) -> BackendStatus {
        BackendStatus {
            workspace_root: self.services.config.workspace_root.clone(),
            capabilities: self.services.capabilities.clone(),
            connectors: self.connectors.statuses().await,
            tools: self.tools.specs(),
            model_roles: self.services.role_registry.all(),
            runtime: self.runtime_state(),
        }
    }
}
