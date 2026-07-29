use futures::future::BoxFuture;
use hawking_orch::inference::InferenceClient;
use hawking_orch::router::SimpleRouter;
use hide_backend::{BackendHost, MemoryScope};
use hide_core::api::Intent;
use hide_core::event::Event;
use hide_core::ids::{now_ms, RunId, SessionId};
use hide_core::runtime::{GenerationStats, InferenceRequest, StreamChunk, TokenSink};
use hide_core::Result;
use hide_kernel::govern::{Autonomy, Interrupt};
use hide_kernel::machine::state::Phase;
use hide_kernel::plan::planner::Planner;
use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
use hide_kernel::runtime_client::KernelRuntimeClient;
use hide_kernel::AgentKernel;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
fn test_host() -> BackendHost {
    static N: AtomicU64 = AtomicU64::new(0);
    let uniq = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("hide_steer_{}_{}", now_ms(), uniq));
    BackendHost::open_workspace(&dir).unwrap()
}
struct CapturingInferenceClient {
    prompts: parking_lot::Mutex<Vec<String>>,
}
impl CapturingInferenceClient {
    fn new() -> Self {
        Self {
            prompts: parking_lot::Mutex::new(Vec::new()),
        }
    }
}
impl InferenceClient for CapturingInferenceClient {
    fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        self.prompts.lock().push(request.prompt.clone());
        Box::pin(async move {
            sink(StreamChunk::Token {
                token_id: None,
                text: "ok".to_string(),
            })?;
            sink(StreamChunk::Done {
                reason: "stop".to_string(),
                stats: None,
            })?;
            Ok(GenerationStats {
                input_tokens: 0,
                output_tokens: 1,
                decode_tokens_per_second: None,
            })
        })
    }
    fn embed<'a>(&'a self, _text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        Box::pin(async move { Ok(vec![0.0; 8]) })
    }
}
struct TwoReadsPlanner;
impl Planner for TwoReadsPlanner {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>> {
        let objective = objective.to_string();
        Box::pin(async move {
            let read1 = PlanStep::new(
                "read the first file",
                StepKind::Investigate,
                Acceptance::predicate("first file understood"),
            );
            let mut read2 = PlanStep::new(
                "read the second file",
                StepKind::Investigate,
                Acceptance::predicate("second file understood"),
            );
            read2.dependencies = vec![read1.id.clone()];
            Ok(Plan {
                id: hide_core::ids::PlanId::new(),
                title: "two reads".to_string(),
                objective,
                steps: vec![read1, read2],
                status: PlanStatus::Active,
                budget: Default::default(),
            })
        })
    }
}
async fn observation_count(host: &BackendHost, session: &SessionId) -> usize {
    host.services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap()
        .iter()
        .filter(|e: &&Event| e.kind == "agent.observation")
        .count()
}
#[tokio::test]
async fn trace_a_steer_reaches_running_kernel_and_folds_into_next_step() {
    const STEER: &str = "STOP reading, switch to the auth module instead";
    let host = test_host();
    let session = host.services.session();
    let capturing = Arc::new(CapturingInferenceClient::new());
    let runtime = Arc::new(KernelRuntimeClient::new(
        Arc::new(SimpleRouter::new(host.services.role_registry.clone())),
        capturing.clone(),
    ));
    let kernel = AgentKernel::builder(host.services.event_log.clone())
        .autonomy(Autonomy::FullAuto)
        .planner(Arc::new(TwoReadsPlanner))
        .runtime(runtime)
        .build();
    let host_run = RunId::new();
    let mut state = kernel
        .start_run(session.clone(), "investigate the codebase")
        .await
        .unwrap();
    let mut steered = false;
    let mut steer_delivered = false;
    for _ in 0..64 {
        if let Some(Interrupt::Steer { .. }) =
            host.interrupts().drain_into_kernel(&host_run, &kernel)
        {
            steer_delivered = true;
        }
        if state.phase.is_terminal() {
            break;
        }
        kernel.step(&mut state).await.unwrap();
        if !steered && observation_count(&host, &session).await >= 1 {
            let ack = host
                .handle_intent(Intent::Custom {
                    name: "redirect_run".to_string(),
                    payload: serde_json::json!({
                        "run_id": host_run.as_str(),
                        "text": STEER,
                        "session_id": session.as_str(),
                    }),
                })
                .await
                .unwrap();
            assert!(ack.accepted, "the steer intent is accepted");
            steered = true;
        }
    }
    assert!(steer_delivered, "InterruptHub forwarded a Steer to the kernel");
    assert!(observation_count(&host, &session).await >= 2);
    let prompts = capturing.prompts.lock().clone();
 assert!( prompts.len() >= 2, "both read steps generated a prompt: {prompts:?}" );
    assert!(!prompts[0].contains(STEER));
    assert!(prompts.iter().skip(1).any(|p| p.contains(STEER)));
    let events = host
        .services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap();
    let steer_event = events
        .iter()
        .find(|e: &&Event| e.kind == "turn.steer")
        .expect("a durable turn.steer event is persisted");
    assert_eq!(steer_event.payload.get("instruction").and_then(|v| v.as_str()), Some(STEER));
    assert_eq!(steer_event.run_id.as_ref().map(|r| r.as_str()), Some(host_run.as_str()));
    assert_eq!(state.phase, Phase::Done, "the steered run still completed");
}
#[tokio::test]
async fn memory_add_intent_persists_a_record() {
    let host = test_host();
    let scope = MemoryScope::Repo("hawking".to_string());
    let ack = host
        .handle_intent(Intent::Custom {
            name: "memory_add".to_string(),
            payload: serde_json::json!({
                "scope": { "kind": "repo", "id": "hawking" },
                "claim": "the turn loop is a single flat FSM",
                "source": "census",
                "author": "tester",
                "citations": ["crates/hide-kernel/src/machine/driver.rs"],
            }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    let records = host.memory_list(&scope);
    assert_eq!(records.len(), 1, "one memory record was persisted");
    assert_eq!(records[0].claim, "the turn loop is a single flat FSM");
    assert!(host.memory_get(&records[0].memory_id).is_some());
}
#[tokio::test]
async fn goal_evaluate_intent_returns_a_deterministic_verdict() {
    use hide_backend::GoalStatus;
    use hide_core::event::NewEvent;
    use hide_kernel::verify::oracle::{OracleClass, Verdict};
    let host = test_host();
    let session = host.services.session();
    host.goal_set(session.clone(), "tests_pass", vec!["tests".to_string()])
        .unwrap();
    host.services
        .event_log
        .append(NewEvent::system(
            session.clone(),
            "verify.result",
            serde_json::to_value(&Verdict::pass("tests", OracleClass::Deterministic, "all green"))
                .unwrap(),
        ))
        .await
        .unwrap();
    let ack = host
        .handle_intent(Intent::Custom {
            name: "goal_evaluate".to_string(),
            payload: serde_json::json!({ "session_id": session.as_str() }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    assert_eq!(host.goal_get(&session).unwrap().status, GoalStatus::Met);
}
#[tokio::test]
async fn workspace_set_repo_trust_intent_is_held_until_approved() {
    use hide_backend::services::{RepoNode, TrustState};
    let host = test_host();
    host.workspace_add_repo(RepoNode::new("vendor", "/tmp/vendor"))
        .unwrap();
    assert_eq!(host.workspace_repo("vendor").unwrap().trust, TrustState::Untrusted);
    let ack = host
        .handle_intent(Intent::Custom {
            name: "workspace_set_repo_trust".to_string(),
            payload: serde_json::json!({ "repo_id": "vendor", "trust": "trusted" }),
        })
        .await
        .unwrap();
    assert!(ack.accepted, "the intent is recorded");
    let message = ack.message.expect("an Ask command reports its gate");
    let gate = message
        .split("gate=")
        .nth(1)
        .expect("the ack names the gate id")
        .to_string();
    assert_eq!(host.workspace_repo("vendor").unwrap().trust, TrustState::Untrusted);
    host.handle_intent(Intent::Custom {
        name: "approve_gate".to_string(),
        payload: serde_json::json!({ "gate": gate }),
    })
    .await
    .unwrap();
    assert_eq!(host.workspace_repo("vendor").unwrap().trust, TrustState::Trusted);
}
#[tokio::test]
async fn environment_switch_intent_emits_event_and_updates_current_env() {
    use hide_backend::services::{EnvironmentNode, WorkspaceStore};
    let host = test_host();
    let session = host.services.session();
    host.workspace_add_environment(EnvironmentNode::new("container:node20"))
        .unwrap();
    let ack = host
        .handle_intent(Intent::Custom {
            name: "environment_switch".to_string(),
            payload: serde_json::json!({
                "session_id": session.as_str(),
                "env_id": "container:node20",
                "reason": "run the node build",
            }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    let switches = host.environment_switches(&session).await.unwrap();
    assert_eq!(switches.len(), 1, "one environment switch was recorded");
    assert_eq!(switches[0].new_env, "container:node20");
    assert_eq!(WorkspaceStore::current_env(&host.services.key_value_store, &session).as_deref(), Some("container:node20"));
}
