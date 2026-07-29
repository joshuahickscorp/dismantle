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

    /// Spawn the generation for an accepted `SubmitTurn`: route it at the live
    /// runtime and stream tokens onto Wire-B. The run's `run_id` is registered
    /// so `CancelRun`/`PauseRun` reach it via the shared `InterruptHub`. When no
    /// runtime is online (no model configured, or it is not yet `Ready`), this
    /// publishes a `RuntimeStatus`/`Error` UiEvent instead of generating, so the
    /// FE shows "model offline", never a fake token.
    pub(crate) fn spawn_submit_turn_generation(&self, session_id: SessionId, prompt: String) {
        let run_id = RunId::new();
        match self.runtime_base_url() {
            Some(base_url) => {
                // Register the run with the interrupt hub so control intents can
                // reach it (the generation task polls it cooperatively).
                let ui_bus = self.ui_bus.clone();
                let role_registry = self.services.role_registry.clone();
                let event_log = self.services.event_log.clone();
                let key_value_store = self.services.key_value_store.clone();
                let code_index = self.services.code_index.clone();
                let memory = self.services.memory_store.clone();
                let classed_memory = self.services.classed_memory.clone();
                let interrupts = self.interrupts.clone();
                let approvals = self.approvals.clone();
                let run = run_id.clone();
                let repo_instructions = self.services.repo_instructions.clone();
                if kernel_turn_enabled() {
                    // Increment 2: route the turn through the REAL kernel loop so
                    // it can plan, use tools (permission-gated), and verify with
                    // deterministic oracles. The kernel is built here (sync) via
                    // `build_turn_kernel` and moved into the spawned driver.
                    let kernel =
                        self.build_turn_kernel(base_url.clone(), session_id.clone(), run.clone());
                    tokio::spawn(async move {
                        if let Err(e) = run_turn_kernel(
                            kernel,
                            event_log,
                            key_value_store,
                            role_registry,
                            code_index,
                            memory,
                            classed_memory,
                            ui_bus.clone(),
                            interrupts,
                            approvals,
                            run,
                            session_id.clone(),
                            base_url,
                            prompt,
                            DEFAULT_KERNEL_TURN_MAX_STEPS,
                            repo_instructions,
                        )
                        .await
                        {
                            ui_bus.publish(UiEvent {
                                seq: 0,
                                session_id: Some(session_id),
                                kind: UiEventKind::Error {
                                    code: "generation".to_string(),
                                    message: e.to_string(),
                                },
                            });
                        }
                    });
                } else {
                    // Single-shot fallback (no tools): the model-offline-safe path,
                    // pinnable via `HIDE_KERNEL_TURN=0` for tests / degraded serves.
                    tokio::spawn(async move {
                        if let Err(e) = generate_submit_turn(
                            event_log,
                            role_registry,
                            code_index,
                            memory,
                            classed_memory,
                            ui_bus.clone(),
                            interrupts,
                            run,
                            session_id.clone(),
                            base_url,
                            prompt,
                            repo_instructions,
                        )
                        .await
                        {
                            // Surface the failure on the same typed Wire-B channel;
                            // never swallow it.
                            ui_bus.publish(UiEvent {
                                seq: 0,
                                session_id: Some(session_id),
                                kind: UiEventKind::Error {
                                    code: "generation".to_string(),
                                    message: e.to_string(),
                                },
                            });
                        }
                    });
                }
            }
            None => {
                // No model online: surface "model offline" as a real UiEvent.
                let status = self
                    .runtime_state()
                    .map(|s| format!("{s:?}").to_lowercase())
                    .unwrap_or_else(|| "down".to_string());
                let detail = match self.runtime.is_some() {
                    true => "runtime not ready; reconnect when it reports ready".to_string(),
                    false => "no model configured (set HIDE_MODEL_WEIGHTS)".to_string(),
                };
                self.ui_bus.publish(UiEvent {
                    seq: 0,
                    session_id: Some(session_id),
                    kind: UiEventKind::RuntimeStatus {
                        status,
                        detail: Some(detail),
                    },
                });
            }
        }
    }

    pub async fn call_connector(&self, id: &str, method: &str, params: Value) -> Result<Value> {
        self.connectors.call(id, method, params).await
    }

    pub async fn rebuild_session_projection(
        &self,
        session_id: SessionId,
    ) -> Result<SessionProjection> {
        self.replay.rebuild_session(session_id).await
    }

    pub async fn ui_events(
        &self,
        session_id: Option<SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<UiEvent>> {
        self.replay.ui_events(session_id, after_seq, limit).await
    }

    pub async fn run_command(
        &self,
        session_id: SessionId,
        argv: Vec<String>,
        cwd: Option<String>,
    ) -> Result<ToolResult> {
        let mut args = json!({ "argv": argv });
        if let Some(cwd) = cwd {
            args["cwd"] = json!(cwd);
        }
        self.dispatch_tool(session_id, None, ToolCall::new("shell.run", args))
            .await
    }

    // -- Session-aware terminal process surface (Trace D) ------------------
    //
    // The terminal is a supervised, sandbox-confined process surface. These are
    // the host-level handles the FE (or a headless caller) drives: start a
    // (possibly long-lived) process, let it keep running across navigation,
    // attach/detach its streamed output, stop it, capture its logs as a durable
    // artifact, and read its compact state.

    /// The process supervisor (for inspection / advanced callers).
    pub fn processes(&self) -> &Arc<ProcessSupervisor> {
        &self.processes
    }

    /// Start a managed terminal process, sandbox-confined. `persistent` keeps it
    /// running independent of any session/turn; `owner` records the owning run or
    /// job. Returns the process id. A spawn fault or fail-closed sandbox refusal is
    /// recorded as a `failed` process (queryable via [`BackendHost::process_state`]).
    pub fn start_process(
        &self,
        argv: Vec<String>,
        cwd: Option<String>,
        env: std::collections::BTreeMap<String, String>,
        persistent: bool,
        owner: Option<String>,
    ) -> String {
        let spec = StartSpec {
            argv,
            cwd,
            env,
            persistent,
            owner,
            interactive: persistent,
        };
        self.processes.start(spec, &self.shell_config())
    }

    /// Whether a managed process is still alive.
    pub fn process_alive(&self, id: &str) -> bool {
        self.processes.is_alive(id)
    }

    /// A compact snapshot of a managed process (env, cwd, status, exit, sandboxed
    /// flag, owner), or `None` if the id is unknown.
    pub fn process_state(&self, id: &str) -> Option<ProcessState> {
        self.processes.state(id)
    }

    /// Attach a turn to a running process: replay its buffered output onto the bus
    /// under `session` and resume live mirroring. Returns the buffered lines.
    pub fn attach_process(&self, id: &str, session: SessionId) -> Option<Vec<String>> {
        self.processes.attach(id, session)
    }

    /// Detach the live UI mirror; the process keeps running and buffering.
    pub fn detach_process(&self, id: &str) -> bool {
        self.processes.detach(id)
    }

    /// Stop a managed process (SIGTERM the group, then SIGKILL after a grace).
    pub fn stop_process(&self, id: &str) -> bool {
        self.processes.stop(id)
    }

    /// Preserve a process's captured output as a durable blob-store artifact.
    pub fn capture_process_artifact(&self, id: &str) -> Result<hide_core::types::BlobRef> {
        match self.processes.capture_artifact(id, &self.services.blob_store) {
            Some(res) => res,
            None => Err(hide_core::error::HideError::NotFound(format!(
                "unknown process {id}"
            ))),
        }
    }

    /// Deliver a terminal intent to the targeted managed process. Returns `Err(reason)` for the
    /// caller to surface as an Error UiEvent (and a refused ack).
    ///
    /// * `pty_input` / `pty_resize`: write stdin / record geometry. `process` is optional; absent =
    ///   the most recently started live process.
    /// * `attach_process` / `stop_process` / `capture_process_artifact`: the three process controls
    ///   that had no wire trigger at all, so a client could START a sandboxed process and then had
    ///   no way to attach to it after navigating away, stop it, or keep its output. Not being able
    ///   to stop what you started is the safety half of that. `process` is REQUIRED here: these
    ///   address one named process, and guessing "the latest" would stop the wrong one.
    pub(crate) async fn handle_process_intent(
        &self,
        name: &str,
        payload: &Value,
    ) -> std::result::Result<(), String> {
        let id = payload.get("process").and_then(|v| v.as_str());
        let named = || id.ok_or_else(|| format!("{name}: missing 'process'"));
        match name {
            "pty_input" => {
                let data = payload
                    .get("data")
                    .or_else(|| payload.get("text"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "pty_input: missing 'data'".to_string())?;
                self.processes.write_stdin(id, data).await
            }
            "pty_resize" => {
                let cols = payload.get("cols").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
                let rows = payload.get("rows").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
                self.processes.resize(id, cols, rows)
            }
            // Re-attaching replays the buffered output onto the bus under the attaching session, so
            // a re-navigated terminal comes back with its scrollback instead of an empty pane.
            "attach_process" => {
                let id = named()?;
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(SessionId::from)
                    .unwrap_or_else(|| self.services.session());
                let lines = self
                    .attach_process(id, session.clone())
                    .ok_or_else(|| format!("unknown process {id}"))?;
                self.publish_custom(
                    Some(session),
                    json!({ "kind": "process_attached", "process": id, "lines": lines }),
                );
                Ok(())
            }
            "stop_process" => {
                let id = named()?;
                if !self.stop_process(id) {
                    return Err(format!("unknown process {id}"));
                }
                self.publish_custom(None, json!({ "kind": "process_stopped", "process": id }));
                Ok(())
            }
            "capture_process_artifact" => {
                let id = named()?;
                let blob = self.capture_process_artifact(id).map_err(|e| e.to_string())?;
                self.publish_custom(
                    None,
                    json!({
                        "kind": "process_artifact",
                        "process": id,
                        "artifact": serde_json::to_value(&blob).unwrap_or(Value::Null),
                    }),
                );
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Publish a seq-0 `Custom` UiEvent on the live bus.
    pub(crate) fn publish_custom(&self, session_id: Option<SessionId>, data: Value) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id,
            kind: UiEventKind::Custom(data),
        });
    }

    /// The run a session's editor saves are grouped under, so every save lands on ONE addressable
    /// [`DiffProposal`] (`diff-editor-<session>`) instead of a diff per keystroke-save. Stable per
    /// session and derived, not stored, so a restart addresses the same diff.
    pub fn editor_run(session: &SessionId) -> RunId {
        RunId::from(format!("editor-{}", session.as_str()))
    }

    /// The editor save (`{ path, content, base_hash?, session_id? }`), and the ONE wire-reachable
    /// workspace write.
    ///
    /// It goes through [`Self::dispatch_tool`] WITH a run id, not through the `fs` connector's
    /// dispatcher call, because `dispatch_tool` is where the whole downstream chain hangs off: the
    /// `tool.call`/`tool.result` pair the timeline and transcript search read, and the
    /// `record_edit_diff` capture the hunk review surface, the checkpoint's `repo_state` coverage
    /// and the code rewind all read. Routing straight at the dispatcher applied the bytes and fed
    /// none of them, so the app could write files that no consumer could see, review or undo.
    ///
    /// The session is the caller's (the FE's `runCommand` fills `session_id` into every custom
    /// payload); a payload without one falls back to the default session, as the other
    /// session-scoped intents do.
    pub(crate) async fn save_file_effect(&self, payload: &Value) -> Result<()> {
        let path = payload
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| hide_core::error::HideError::Config("missing path".to_string()))?;
        let content = payload
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| hide_core::error::HideError::Config("missing content".to_string()))?;
        // Same confinement the connector read path uses: `..`, absolute and prefix components are
        // refused before the dispatcher ever sees the call.
        let abs = crate::connectors::workspace_resolve(&self.services.config.workspace_root, path)?;
        let mut args = json!({ "path": abs.to_string_lossy(), "content": content });
        // `base_hash` (the blake3 of the text the caller read) is passed through when supplied, so
        // a concurrently-changed file CONFLICTS instead of being clobbered.
        if let Some(base) = payload.get("base_hash").and_then(|v| v.as_str()) {
            args["base_hash"] = json!(base);
        }
        let session = payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(SessionId::from)
            .unwrap_or_else(|| self.services.session());
        let run = Self::editor_run(&session);
        let result = self
            .dispatch_tool(session, Some(run), ToolCall::new("edit.write_file", args))
            .await?;
        // `Tool`, not `PolicyDenied`: the permission verdict is the `?` above. This is the applier
        // refusing (a `base_hash` conflict, most often), which no approval fixes, so it must not be
        // mistaken for something to hold at a gate.
        if result.status != ToolStatus::Ok {
            return Err(hide_core::error::HideError::Tool(format!(
                "write of {path} refused: {}",
                result
                    .error
                    .as_ref()
                    .map(|e| format!("{}: {}", e.code, e.message))
                    .unwrap_or_else(|| "rejected by the applier".to_string())
            )));
        }
        Ok(())
    }

    /// Dispatch a tool ATTRIBUTED to a session and (optionally) a run.
    ///
    /// A thin caller of the recorded path: the durable `tool.call`/`tool.result` pair, the live
    /// `ToolProgress`, and the reviewable/revertible diff an `edit.*` write produces are recorded
    /// by [`DispatchRecorder`] hanging off the dispatcher itself, so the kernel agent - which holds
    /// the dispatcher directly and never enters this function - records exactly the same things.
    /// All this adds is the attribution: without it a dispatch is recorded against the default
    /// session with no run, which is honest but ungrouped.
    pub async fn dispatch_tool(
        &self,
        session_id: SessionId,
        run_id: Option<RunId>,
        call: ToolCall,
    ) -> Result<ToolResult> {
        crate::tools::with_dispatch_context(session_id, run_id, self.dispatcher.dispatch(call))
            .await
    }

    /// Schedule a parallel kernel run via `hide_fleet::FleetManager` and drive it
    /// to completion (the now-real fleet path - the previously-dead `hide-fleet`
    /// dep is load-bearing here). The run is enqueued, admitted under the fleet
    /// Governor, isolated in a **real git worktree** (one per agent, never shared,
    /// released on completion), and driven by a `KernelRunLauncher` over a
    /// [`RuntimePlanner`]-wired kernel. Returns the job's terminal status string.
    ///
    /// The launcher kernel is built on-demand via [`build_fleet_kernel`]: a
    /// `RuntimePlanner` (model plans when Ready, else falls back to the canonical
    /// investigate→edit→verify DAG) — never `AgentKernel::new` / StubPlanner.
    pub async fn fleet_run(
        &self,
        session_id: SessionId,
        objective: impl Into<String>,
    ) -> Result<String> {
        // Real OS probe (free RAM + thermal proxy). Replaces the prior canned
        // FixedResourceProbe { free_memory_mb: 32_768 } so discovery reports
        // this machine, not a fake 32 GiB. Tests that need a fixed envelope
        // still inject FixedResourceProbe at the FleetManager constructor.
        let probe = Arc::new(OsResourceProbe::default());
        let kernel = Arc::new(self.build_fleet_kernel(session_id.clone()));
        let launcher = Arc::new(KernelRunLauncher::new(kernel).with_max_steps(128));
        let repo_root = self
            .services
            .config
            .workspace_root
            .to_string_lossy()
            .to_string();
        // Real git worktrees under the workspace — never `.with_fake_worktrees()`.
        // Isolation fails honestly if the workspace is not a git checkout.
        let manager = FleetManager::new(
            self.services.event_log.clone(),
            FleetGovernor::default(),
            probe,
            launcher,
            FleetConfig {
                repo_root,
                ..FleetConfig::default()
            },
        );

        let job = AgentJob::new(objective, PriorityClass::Normal)
            .with_session(session_id)
            .with_concurrency_class(ConcurrencyClass::Model);
        let job_id = job.id.clone();
        manager.enqueue(job).await?;
        manager.run_to_quiescence(2, 128).await?;

        let status = manager
            .queue()
            .get(&job_id)
            .map(|j| format!("{:?}", j.status))
            .unwrap_or_else(|| "Unknown".to_string());
        Ok(status)
    }

    /// Kernel for the fleet launcher path: **RuntimePlanner** (not StubPlanner),
    /// with a live model when Ready and a stub that yields the default DAG
    /// offline. Standard process oracles + permission-gated dispatcher so plan
    /// steps that name typecheck/build/test are real, not faith.
    pub fn build_fleet_kernel(&self, session_id: SessionId) -> AgentKernel {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};
        use hawking_orch::inference::{InferenceClient, StubInferenceClient};
        use hawking_orch::router::SimpleRouter;
        use hide_kernel::runtime_client::KernelRuntimeClient;

        let inference: Arc<dyn InferenceClient> = if let Some(url) = self.runtime_base_url() {
            Arc::new(ModelProviderInferenceClient::new(HttpModelProvider::new(
                url,
            )))
        } else {
            // Offline: RuntimePlanner.synthesize falls through to default_dag
            // on empty/failed generation — still a real plan, not StubPlanner.
            Arc::new(StubInferenceClient::new(""))
        };
        let runtime = Arc::new(KernelRuntimeClient::new(
            Arc::new(SimpleRouter::new(self.services.role_registry.clone())),
            inference,
        ));
        let dispatcher = self.build_turn_dispatcher(session_id, None);
        AgentKernel::builder(self.services.event_log.clone())
            .workspace_root(
                self.services
                    .config
                    .workspace_root
                    .to_string_lossy()
                    .to_string(),
            )
            // Fleet has no interactive approver on this path: FullAuto so the
            // RuntimePlanner DAG can progress. Oracles still gate correctness.
            .autonomy(Autonomy::FullAuto)
            // `.runtime(..)` installs RuntimePlanner when no planner is set.
            .runtime(runtime)
            .dispatcher(dispatcher.clone())
            .with_standard_oracles(dispatcher)
            .build()
    }

    /// Generate against a (supervised) runtime through the kernel's runtime-client
    /// seam and publish the completion onto the push Wire-B bus.
    ///
    /// This is the host's end-to-end generation path: a `KernelRuntimeClient`
    /// (router + the host's HTTP `ModelProvider`, adapted to the orch
    /// `InferenceClient` seam) produces tokens; each token batch is published - 
    /// with coalescing - onto the broadcast bus, then flushed at stream end. The
    /// returned string is the full completion (for callers that also want it
    /// inline). `base_url` is the supervised serve's base (from the
    /// `RuntimeSupervisor`).
    pub async fn generate_and_publish(
        &self,
        session_id: SessionId,
        base_url: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<String> {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};

        let provider = HttpModelProvider::new(base_url);
        let inference: Arc<dyn hawking_orch::inference::InferenceClient> =
            Arc::new(ModelProviderInferenceClient::new(provider));
        // Both generation entry points funnel through `run_turn_core` so the live
        // path and this one build the SAME real request (compiled context + real
        // history + a derived budget) and can never drift. This twin skips the
        // per-step / post-turn live-manifest telemetry (no run/interrupt wiring).
        let outcome = run_turn_core(
            inference,
            self.services.event_log.clone(),
            self.services.role_registry.clone(),
            self.services.code_index.clone(),
            self.services.memory_store.clone(),
            self.services.classed_memory.clone(),
            self.ui_bus.clone(),
            session_id,
            prompt.into(),
            None,
            None,
            self.services.repo_instructions.clone(),
        )
        .await?;
        Ok(outcome.completion)
    }

    /// Time-travel: scrub a session's projection to (and including) `seq`. A
    /// read-only view into the past (does not clobber the live projection).
    pub async fn scrub_to_event(
        &self,
        session_id: SessionId,
        seq: u64,
    ) -> Result<SessionProjection> {
        self.replay.scrub_to_event(session_id, seq).await
    }

    /// Time-travel: fork a new session from `from`'s log prefix up to `at_seq`.
    pub async fn fork_session(
        &self,
        from: SessionId,
        at_seq: u64,
    ) -> Result<(SessionId, SessionProjection)> {
        self.replay.fork_session(from, at_seq).await
    }

    /// Time-travel FORK by EVENT boundary (bible sec 78.1 #7): create a NEW
    /// session whose durable history is `from` folded up to (and including)
    /// `at_event`, with ANCESTRY recorded (parent + boundary) and the new thread
    /// SURFACED to the client as a [`SessionRecord`] plus a UiEvent. `at_event =
    /// None` forks the WHOLE session (its current tail).
    ///
    /// Independence is structural: [`BackendReplayService::fork_session`]
    /// re-appends the source prefix under a fresh `SessionId` (a new event
    /// lineage), so the original is untouched and later appends to either side
    /// never cross over. Ancestry is stored OUT of the fork's own event log (in
    /// the KV `session_records` namespace) so it never pollutes the fork's
    /// transcript and survives a workspace reopen.
    pub async fn fork_session_from_event(
        &self,
        from: SessionId,
        at_event: Option<&hide_core::ids::EventId>,
    ) -> Result<(SessionId, crate::services::SessionRecord, SessionProjection)> {
        let (new_session, record, projection) = fork_and_record(
            &self.replay,
            &self.services.sessions,
            &self.services.key_value_store,
            from,
            at_event.cloned(),
        )
        .await?;
        // Surface the new thread to the client: a durable record + a live UiEvent,
        // published UNDER the new session id so the FE adopts the fork.
        self.publish_session_forked(&new_session, &record);
        Ok((new_session, record, projection))
    }

    /// Search the durable transcript (bible sec 32-33): a LITERAL substring plus
    /// STRUCTURED filters (kind / session / role / time range), ranked
    /// deterministically and bounded. No model, no embeddings (semantic search is
    /// `DEFERRED_MODEL_REQUIRED`).
    pub async fn search_transcript(
        &self,
        query: &crate::replay::TranscriptQuery,
    ) -> Result<Vec<crate::replay::TranscriptHit>> {
        self.replay.search_transcript(query).await
    }

    /// Dispatch a `search` / `search_transcript` custom intent (census sec 32-33):
    /// build a [`TranscriptQuery`](crate::replay::TranscriptQuery) from the FE
    /// payload and run the model-free literal + structured search. The command
    /// palette speaks `/intent`, so this is how it searches without a `/rpc` dial.
    ///
    /// Payload: `{ query | text, session_id?, kind?, role?, since_ts?, until_ts?,
    /// limit?, scopes? }`. The structured filters (`kind` / `role`) may sit at the
    /// top level or under a `scopes` object (top level wins). Semantic search is
    /// DEFERRED_MODEL_REQUIRED and never runs here.
    pub(crate) async fn handle_search_intent(
        &self,
        payload: &Value,
    ) -> Result<Vec<crate::replay::TranscriptHit>> {
        let scopes = payload.get("scopes");
        // A field is read from the top level, falling back to `scopes`.
        let field = |name: &str| {
            payload
                .get(name)
                .or_else(|| scopes.and_then(|s| s.get(name)))
        };
        let mut query = crate::replay::TranscriptQuery {
            text: payload
                .get("query")
                .or_else(|| payload.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            ..Default::default()
        };
        query.session_id = field("session_id")
            .and_then(|v| v.as_str())
            .map(SessionId::from);
        query.kind = field("kind").and_then(|v| v.as_str()).map(str::to_string);
        query.role = field("role").and_then(|v| v.as_str()).map(str::to_string);
        query.since_ts = field("since_ts").and_then(|v| v.as_u64());
        query.until_ts = field("until_ts").and_then(|v| v.as_u64());
        query.limit = payload
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        self.search_transcript(&query).await
    }

    /// Surface transcript-search hits to the FE as a `search_results` UiEvent
    /// (echoing the query so the palette can correlate the response).
    pub(crate) fn publish_search_results(&self, payload: &Value, hits: &[crate::replay::TranscriptHit]) {
        let query = payload
            .get("query")
            .or_else(|| payload.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: None,
            kind: UiEventKind::Custom(json!({
                "kind": "search_results",
                "query": query,
                "count": hits.len(),
                "hits": serde_json::to_value(hits).unwrap_or_else(|_| json!([])),
            })),
        });
    }

    /// The conversation graph (bible sec 32-33) rooted at `session_id`: the node,
    /// its ancestry chain (to a root), and its direct children (forks / side chats
    /// / ephemeral forks), with parent->child edges. A bounded, DETERMINISTIC
    /// projection over the durable `session_records` KV -- no model, safe headless.
    pub fn conversation_graph(&self, session_id: &SessionId) -> crate::services::ConversationGraph {
        self.services
            .sessions
            .conversation_graph(&self.services.key_value_store, session_id)
    }

    // --- Multi-repo workspace graph (bible sec 35, sec 78.1 #14) -------------

    /// Add (or replace) a REPOSITORY node in the workspace graph. Idempotent by
    /// `repo_id`. A repo enters UNTRUSTED unless the node already carries a trust
    /// decision (trust-before-config): while untrusted its instructions / policy
    /// refs are inert (see [`RepoNode::active_instructions_ref`]). Written to the
    /// durable KV `workspace_repos` namespace so the graph survives a reopen.
    pub fn workspace_add_repo(&self, repo: RepoNode) -> Result<RepoNode> {
        WorkspaceStore::put_repo(&self.services.key_value_store, &repo)?;
        Ok(repo)
    }

    /// Look up a repo node by id.
    pub fn workspace_repo(&self, repo_id: &str) -> Option<RepoNode> {
        WorkspaceStore::get_repo(&self.services.key_value_store, repo_id)
    }

    /// TRUST (or untrust) a repo already in the graph: the trust-before-config
    /// gate. Only after this flips to `Trusted` are the repo's instructions /
    /// policy refs treated active. Returns the updated node, or `None` when no
    /// such repo exists.
    pub fn workspace_set_repo_trust(
        &self,
        repo_id: &str,
        trust: TrustState,
    ) -> Result<Option<RepoNode>> {
        let kv = &self.services.key_value_store;
        match WorkspaceStore::get_repo(kv, repo_id) {
            Some(mut repo) => {
                repo.trust = trust;
                WorkspaceStore::put_repo(kv, &repo)?;
                Ok(Some(repo))
            }
            None => Ok(None),
        }
    }

    // --- The task-scoped transactional write lease --------------------------
    //
    // Data shape, enforcement point, restart policy: crates/hide-backend/src/tools.rs.
    // The host owns only the GRANT conditions, the REVOCATION triggers, and the read the
    // status bar renders.

    /// Install the write lease for an approved task.
    ///
    /// Reached ONLY from [`Self::released_effect`], i.e. only after a human approved the
    /// `grant_write_lease` gate. That approval is the "user explicitly started or approved an
    /// implementation task" condition, and it is not forgeable from here: the effect is
    /// `ApprovalPolicy::Ask` in the ONE catalog, so [`Self::gated_effect`]'s sibling machinery
    /// parks it whatever channel it arrived on.
    ///
    /// The remaining grant conditions are read, not assumed:
    /// * `repo_id` must name a repo in the workspace graph and that repo must be TRUSTED. An
    ///   untrusted repo is inert by trust-before-config, so it can never be leased.
    /// * the scope is the trusted repo's OWN root, optionally narrowed by declared relative
    ///   sub-paths. `workspace_resolve` is the same confinement helper the fs connector uses, so a
    ///   declared scope cannot name a path outside the repo it claims to be inside.
    ///
    /// `{ repo_id, scopes?: [rel], session_id?, run_id? }`.
    pub(crate) async fn handle_grant_write_lease(&self, payload: &Value) -> Result<()> {
        let repo_id = payload
            .get("repo_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                hide_core::error::HideError::Message(
                    "grant_write_lease: missing 'repo_id'".to_string(),
                )
            })?;
        let repo = self.workspace_repo(repo_id).ok_or_else(|| unknown_repo(repo_id))?;
        if repo.trust != TrustState::Trusted {
            return Err(hide_core::error::HideError::PolicyDenied(format!(
                "{repo_id} is not trusted; trust the repository before granting it a write lease"
            )));
        }
        let scopes = match payload.get("scopes").and_then(|v| v.as_array()) {
            Some(rels) if !rels.is_empty() => rels
                .iter()
                .filter_map(|v| v.as_str())
                .map(|rel| crate::connectors::workspace_resolve(&repo.root_path, rel))
                .collect::<Result<Vec<_>>>()?,
            _ => vec![repo.root_path.clone()],
        };
        let lease = crate::tools::install_write_lease(crate::tools::WriteLease {
            lease_id: hide_core::ids::GrantId::new().as_str().to_string(),
            repo_id: repo_id.to_string(),
            session_id: payload
                .get("session_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            run_id: payload
                .get("run_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            scopes,
            granted_ms: hide_core::ids::now_ms(),
        });
        publish_write_lease(&self.ui_bus, Some(&lease), "granted");
        Ok(())
    }

    /// The lease in force, if any (test / inspection / the status read).
    pub fn write_lease(&self) -> Option<crate::tools::WriteLease> {
        crate::tools::active_write_lease()
    }

    /// Add (or replace) an ENVIRONMENT node in the workspace graph. Idempotent by
    /// `env_id`. Written to the durable KV `workspace_environments` namespace.
    pub fn workspace_add_environment(&self, env: EnvironmentNode) -> Result<EnvironmentNode> {
        WorkspaceStore::put_environment(&self.services.key_value_store, &env)?;
        Ok(env)
    }

    /// Look up an environment node by id.
    pub fn workspace_environment(&self, env_id: &str) -> Option<EnvironmentNode> {
        WorkspaceStore::get_environment(&self.services.key_value_store, env_id)
    }

    /// Add (or replace) a typed EDGE between two repos. Idempotent by the
    /// `from|kind|to` triple. Written to the durable KV `workspace_edges`
    /// namespace.
    pub fn workspace_add_edge(
        &self,
        from: impl Into<String>,
        to: impl Into<String>,
        kind: WorkspaceEdgeKind,
    ) -> Result<WorkspaceEdge> {
        let edge = WorkspaceEdge::new(from, to, kind);
        WorkspaceStore::put_edge(&self.services.key_value_store, &edge)?;
        Ok(edge)
    }

    /// The deterministic multi-repo workspace-graph projection (bible sec 35):
    /// every repo node, every environment node, and every typed edge, each in a
    /// stable order (repos by id, environments by id, edges by from/kind/to). A
    /// flat, model-free read of the durable `workspace_*` KV namespaces.
    pub fn workspace_graph(&self) -> WorkspaceGraph {
        WorkspaceStore::graph(&self.services.key_value_store)
    }

    /// Switch a session's active ENVIRONMENT (bible sec 35.3) WITHOUT losing the
    /// session/thread: the switch is recorded as a durable `environment.switch`
    /// event on the SAME session log, carrying `{ previous_env, new_env, reason,
    /// fs_roots, tool_scopes }`, and the session's current-environment pointer is
    /// advanced. The target environment must already be in the graph (`NotFound`
    /// otherwise). The session id is unchanged and the log keeps growing, so the
    /// caller continues in the same thread under the new context.
    pub async fn environment_switch(
        &self,
        session: SessionId,
        env_id: &str,
        reason: impl Into<String>,
    ) -> Result<EnvironmentSwitch> {
        let kv = &self.services.key_value_store;
        let env = WorkspaceStore::get_environment(kv, env_id).ok_or_else(|| {
            hide_core::error::HideError::NotFound(format!(
                "unknown environment {env_id} (add it to the workspace graph first)"
            ))
        })?;
        let previous_env = WorkspaceStore::current_env(kv, &session);
        let switch = EnvironmentSwitch {
            session_id: session.clone(),
            previous_env,
            new_env: env.env_id.clone(),
            reason: reason.into(),
            fs_roots: env.fs_roots.clone(),
            tool_scopes: env.tool_scopes.clone(),
            switched_ms: hide_core::ids::now_ms(),
        };
        // Durable: append the switch to the SAME session log (the thread is not
        // lost, it is the same lineage one event longer), then advance the
        // session's current-environment pointer.
        self.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "environment.switch",
                serde_json::to_value(&switch).unwrap_or(Value::Null),
            ))
            .await?;
        WorkspaceStore::set_current_env(kv, &session, &env.env_id)?;
        self.publish_environment_switch(&switch);
        Ok(switch)
    }

    /// All durable environment switches recorded for a session, in log order
    /// (bible sec 35.3 reader).
    pub async fn environment_switches(
        &self,
        session: &SessionId,
    ) -> Result<Vec<EnvironmentSwitch>> {
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        Ok(events
            .into_iter()
            .filter(|event| event.kind == "environment.switch")
            .filter_map(|event| event.payload_as::<EnvironmentSwitch>())
            .collect())
    }

    /// Publish an `environment_switch` UiEvent carrying the switch record, under
    /// the switched session (so the FE re-scopes fs roots / tool scopes).
    pub(crate) fn publish_environment_switch(&self, switch: &EnvironmentSwitch) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(switch.session_id.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "environment_switch",
                "record": serde_json::to_value(switch).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    // --- Typed effect ledger + policy decisions (bible sec 40, sec 78.1 #7) ---

    /// Evaluate the durable POLICY for a tool call and record it.
    ///
    /// Looks the tool's DECLARED effects up in the builtin capability registry
    /// (`hide_kernel::extension_registry::build_builtin_tool_registry`, never a hardcoded
    /// table), consults the existing `hide-security` permission engine, derives a
    /// typed [`PolicyDecision`], and RECORDS it as a durable `policy.decision`
    /// event carrying `{ tool, effects, decision, reason }` (sec 40.1). The
    /// derived decision is returned and is readable afterwards via
    /// [`Self::policy_decisions`].
    ///
    /// This is ADDITIVE and MODEL-FREE. The [`ToolDispatcher`] still gates every
    /// call against the permission engine independently; nothing here weakens
    /// that path. A model-assisted policy refinement is `DEFERRED_MODEL_REQUIRED`
    /// (see `crate::policy`).
    pub async fn evaluate_tool_policy(
        &self,
        session: &SessionId,
        tool_id: &str,
        args: &Value,
    ) -> Result<PolicyDecision> {
        let effects = tool_declared_effects(tool_id);
        let verdict = self.permission_verdict_for(tool_id, args);
        let (decision, reason) = derive_policy_decision(&effects, &verdict);
        let record = PolicyDecisionRecord {
            tool: tool_id.to_string(),
            effects: effects.iter().map(|effect| effect.as_str().to_string()).collect(),
            decision,
            reason,
        };
        self.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "policy.decision",
                serde_json::to_value(&record).unwrap_or(Value::Null),
            ))
            .await?;
        Ok(decision)
    }

    /// Build a permission-engine verdict for a tool call, mirroring the
    /// [`ToolDispatcher`] request shape: the tool's advertised capability kind, a
    /// target extracted from the call args, and a risk keyed on the spec's
    /// `destructive` annotation. Consulted by [`Self::evaluate_tool_policy`] for
    /// the write path. Model-free.
    pub(crate) fn permission_verdict_for(
        &self,
        tool_id: &str,
        args: &Value,
    ) -> hide_core::permission::PermissionVerdict {
        use hide_core::permission::{PermissionEngine, PermissionRequest};
        use hide_core::types::RiskLevel;
        let engine = SecurityServices::permission_engine(&self.services.config);
        let spec = self.tools.get(tool_id).map(|tool| tool.spec().clone());
        let capability_kind = spec
            .as_ref()
            .and_then(|s| s.capabilities_required.first().cloned())
            .unwrap_or_else(|| "tool.call".to_string());
        let risk = match spec.as_ref() {
            Some(s) if s.annotations.destructive => RiskLevel::High,
            Some(_) => RiskLevel::Low,
            None => RiskLevel::High,
        };
        let target = policy_target_from_args(tool_id, args);
        engine.evaluate(&PermissionRequest {
            capability_kind,
            target,
            risk,
            effects: Vec::new(),
            grant: None,
        })
    }

    /// All durable policy decisions recorded for a session, in log order (bible
    /// sec 40.1 reader).
    pub async fn policy_decisions(
        &self,
        session: &SessionId,
    ) -> Result<Vec<PolicyDecisionRecord>> {
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        Ok(events
            .into_iter()
            .filter(|event| event.kind == "policy.decision")
            .filter_map(|event| event.payload_as::<PolicyDecisionRecord>())
            .collect())
    }
}
