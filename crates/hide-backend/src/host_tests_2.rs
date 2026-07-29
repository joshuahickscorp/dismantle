    #[tokio::test]
    async fn effectful_kernel_turn_pauses_then_resumes_on_approve() {
        let (state, events, target, dir, run_id) =
            drive_effectful_turn(Some(ApprovalDecision::Approve), 64).await;
        assert!(events.iter().any(|e| e.kind == "approval.requested" && e.payload.get("run_id").and_then(|v| v.as_str()) == Some(run_id.as_str())));
        assert!(target.exists(), "approve must let the effectful edit run");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), EFFECT_CONTENT);
        assert_eq!(state.phase, Phase::Done, "approved turn must finish");
        assert!(events.iter().any(|e| e.kind == "approval.resolved" && e.payload.get("decision").and_then(|v| v.as_str()) == Some("approve")));
        assert!(
            events.iter().any(|e| e.kind == "agent.message"
                && e.payload.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && e.payload
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(|t| !t.trim().is_empty())
                    .unwrap_or(false)),
            "approved effectful turn must surface a non-empty assistant answer"
        );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn effectful_kernel_turn_skips_effect_on_deny() {
        let (state, events, target, dir, _run_id) =
            drive_effectful_turn(Some(ApprovalDecision::Deny), 64).await;
        assert!(events.iter().any(|e| e.kind == "approval.requested"));
        assert!(!target.exists(), "deny must skip the effectful edit");
        let step_status = state
            .plan
            .as_ref()
            .and_then(|p| p.steps.first())
            .map(|s| s.status);
        assert_eq!(step_status, Some(hide_kernel::plan::schema::StepStatus::Skipped));
        assert!(state.phase.is_terminal());
        assert!(events.iter().any(|e| e.kind == "approval.resolved" && e.payload.get("decision").and_then(|v| v.as_str()) == Some("deny")));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn effectful_kernel_turn_never_auto_approves_without_a_decision() {
        let (state, events, target, dir, _run_id) = drive_effectful_turn(None, 40).await;
        assert!(events.iter().any(|e| e.kind == "approval.requested"));
 assert!( !target.exists(), "no decision must never auto-apply the effect" );
 assert_ne!( state.phase, Phase::Done, "without approval the turn must not complete the effect" );
        assert!(matches!(state.phase, Phase::Paused | Phase::Aborted));
        assert!(!events.iter().any(|e| e.kind == "approval.resolved"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn approval_request_is_announced_under_the_kind_the_frontend_routes_on() {
        let dir = std::env::temp_dir().join(format!("hide_approval_announce_{}", now_ms()));
        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();
        let ui_bus = Arc::new(UiEventBus::default());
        let mut rx = ui_bus.subscribe();
        let run_id = RunId::new();
        let request = ApprovalRequest {
            step_id: StepId::new(),
            summary: "write src/retry.rs".to_string(),
            effects: vec!["write_fs".to_string()],
        };
        announce_approval_request(&services.event_log, &ui_bus, &session, &run_id, &request)
            .await
            .unwrap();
        let ev = rx.try_recv().expect("the request is pushed on Wire-B");
        let UiEventKind::Custom(v) = ev.kind else {
            panic!("the approval request is a Custom UiEvent")
        };
        assert_eq!(v.get("kind").and_then(|k| k.as_str()), Some("approval_requested"));
        assert_eq!(v.get("run_id").and_then(|k| k.as_str()), Some(run_id.as_str()));
        assert_eq!(v.get("step_id").and_then(|k| k.as_str()), Some(request.step_id.as_str()));
        assert!(v.get("type").is_none(), "no second discriminator to drift");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn approve_effect_intent_deposits_the_decision_into_the_hub() {
        let dir = std::env::temp_dir().join(format!("hide_approve_effect_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let run = RunId::new();
        let step = StepId::new();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "approve_effect".to_string(),
                payload: json!({ "run_id": run.as_str(), "step_id": step.as_str() }),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "approve_effect is recorded + accepted");
        assert_eq!(host.approvals().take(&run), Some((Some(step), ApprovalDecision::Approve)));
        let run2 = RunId::new();
        host.handle_intent(Intent::Custom {
            name: "deny_effect".to_string(),
            payload: json!({ "run_id": run2.as_str() }),
        })
        .await
        .unwrap();
        assert_eq!(host.approvals().take(&run2), Some((None, ApprovalDecision::Deny)));
        let _ = std::fs::remove_dir_all(dir);
    }
    async fn drive_effectful_turn_via_live_intent(
        approve: bool,
    ) -> (AgentState, Vec<hide_core::event::Event>, PathBuf, PathBuf) {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::Duration;
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("hide_live_intent_{}_{}", now_ms(), n));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\nname=\"fx\"\n").unwrap();
        let _ = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .output();
        let host = BackendHost::open_workspace(&dir).unwrap();
        let services = host.services.clone();
        let session = services.session();
        let root = dir.to_string_lossy().to_string();
        let target = dir.join("applied.txt");
        let planner = Arc::new(EditPlanner {
            target: target.to_string_lossy().to_string(),
            content: EFFECT_CONTENT.to_string(),
            oracle: "applied".to_string(),
        });
        let mut suite = hide_kernel::verify::OracleSuite::new();
        suite.register(Arc::new(NoopPassOracle("applied")));
        let dispatcher = hide_kernel::allow_all_dispatcher(root.clone());
        let kernel = AgentKernel::builder(services.event_log.clone())
            .workspace_root(root)
            .autonomy(Autonomy::SuggestOnly)
            .planner(planner as Arc<dyn hide_kernel::plan::planner::Planner>)
            .dispatcher(dispatcher)
            .oracle_suite(suite)
            .build();
        let ui_bus = host.ui_bus().clone();
        let interrupts = host.interrupts().clone();
        let approvals = host.approvals().clone();
        let run_id = RunId::new();
        let event_log = services.event_log.clone();
        let key_value_store = services.key_value_store.clone();
        let role_registry = services.role_registry.clone();
        let code_index = services.code_index.clone();
        let memory = services.memory_store.clone();
        let classed_memory = services.classed_memory.clone();
        let repo_instructions = services.repo_instructions.clone();
        let session_for_turn = session.clone();
        let run_for_turn = run_id.clone();
        let turn = tokio::spawn(async move {
            run_turn_kernel(
                kernel,
                event_log,
                key_value_store,
                role_registry,
                code_index,
                memory,
                classed_memory,
                ui_bus,
                interrupts,
                approvals,
                run_for_turn,
                session_for_turn,
                "http://127.0.0.1:9/unreachable".to_string(),
                "apply the effect".to_string(),
                128,
                repo_instructions,
            )
            .await
        });
        let step_id = {
            let mut found = None;
            for _ in 0..200 {
                let events = services
                    .event_log
                    .scan(Some(session.clone()), None, None)
                    .await
                    .unwrap();
                if let Some(ev) = events.iter().find(|e| {
                    e.kind == "approval.requested"
                        && e.payload.get("run_id").and_then(|v| v.as_str())
                            == Some(run_id.as_str())
                }) {
                    found = ev
                        .payload
                        .get("step_id")
                        .and_then(|v| v.as_str())
                        .map(StepId::from);
                    break;
                }
 assert!( !target.exists(), "effect must not run before the live decision arrives" );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            found.expect("effectful SuggestOnly turn must surface approval.requested")
        };
        let name = if approve {
            "approve_effect"
        } else {
            "deny_effect"
        };
        let ack = host
            .handle_intent(Intent::Custom {
                name: name.to_string(),
                payload: json!({
                    "run_id": run_id.as_str(),
                    "step_id": step_id.as_str(),
                }),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "{name} intent must be accepted");
        let state = turn
            .await
            .expect("turn task joins")
            .expect("run_turn_kernel succeeds");
        let events = services
            .event_log
            .scan(Some(session), None, None)
            .await
            .unwrap();
        (state, events, target, dir)
    }
    #[tokio::test]
    async fn deny_effect_live_intent_skips_effect_and_does_not_hang() {
        let (state, events, target, dir) = drive_effectful_turn_via_live_intent(false).await;
        assert!(events.iter().any(|e| e.kind == "approval.requested"));
 assert!( !target.exists(), "deny_effect via live intent must never run the effectful edit" );
        let step_status = state
            .plan
            .as_ref()
            .and_then(|p| p.steps.first())
            .map(|s| s.status);
        assert_eq!(step_status, Some(hide_kernel::plan::schema::StepStatus::Skipped));
        assert!(state.phase.is_terminal());
        assert!(events.iter().any(|e| e.kind == "approval.resolved" && e.payload.get("decision").and_then(|v| v.as_str()) == Some("deny")));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn approve_effect_live_intent_resumes_runs_effect_and_surfaces_answer() {
        let (state, events, target, dir) = drive_effectful_turn_via_live_intent(true).await;
        assert!(events.iter().any(|e| e.kind == "approval.requested"));
 assert!( target.exists(), "approve_effect via live intent must let the effectful edit run" );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), EFFECT_CONTENT);
        assert_eq!(state.phase, Phase::Done, "approved turn must finish Done");
        assert!(events.iter().any(|e| e.kind == "approval.resolved" && e.payload.get("decision").and_then(|v| v.as_str()) == Some("approve")));
        assert!(
            events.iter().any(|e| e.kind == "agent.message"
                && e.payload.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && e.payload
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(|t| !t.trim().is_empty())
                    .unwrap_or(false)),
            "approved effectful turn must surface a non-empty assistant answer"
        );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn approve_effect_without_step_id_while_nothing_pending_does_not_run_effect() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::Duration;
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("hide_blanket_approve_{}_{}", now_ms(), n));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\nname=\"fx\"\n").unwrap();
        let _ = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .output();
        let host = BackendHost::open_workspace(&dir).unwrap();
        let services = host.services.clone();
        let session = services.session();
        let root = dir.to_string_lossy().to_string();
        let target = dir.join("applied.txt");
        let run_id = RunId::new();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "approve_effect".to_string(),
                payload: json!({ "run_id": run_id.as_str() }),
            })
            .await
            .unwrap();
 assert!( !ack.accepted, "approve_effect without step_id must be refused, got {ack:?}" );
        assert!(!host.approvals().is_pending(&run_id));
        let planner = Arc::new(EditPlanner {
            target: target.to_string_lossy().to_string(),
            content: EFFECT_CONTENT.to_string(),
            oracle: "applied".to_string(),
        });
        let mut suite = hide_kernel::verify::OracleSuite::new();
        suite.register(Arc::new(NoopPassOracle("applied")));
        let dispatcher = hide_kernel::allow_all_dispatcher(root.clone());
        let kernel = AgentKernel::builder(services.event_log.clone())
            .workspace_root(root)
            .autonomy(Autonomy::SuggestOnly)
            .planner(planner as Arc<dyn hide_kernel::plan::planner::Planner>)
            .dispatcher(dispatcher)
            .oracle_suite(suite)
            .build();
        let ui_bus = host.ui_bus().clone();
        let interrupts = host.interrupts().clone();
        let approvals = host.approvals().clone();
        let event_log = services.event_log.clone();
        let key_value_store = services.key_value_store.clone();
        let role_registry = services.role_registry.clone();
        let code_index = services.code_index.clone();
        let memory = services.memory_store.clone();
        let classed_memory = services.classed_memory.clone();
        let repo_instructions = services.repo_instructions.clone();
        let session_for_turn = session.clone();
        let run_for_turn = run_id.clone();
        let turn = tokio::spawn(async move {
            run_turn_kernel(
                kernel,
                event_log,
                key_value_store,
                role_registry,
                code_index,
                memory,
                classed_memory,
                ui_bus,
                interrupts,
                approvals,
                run_for_turn,
                session_for_turn,
                "http://127.0.0.1:9/unreachable".to_string(),
                "apply the effect".to_string(),
                32,
                repo_instructions,
            )
            .await
        });
        let mut saw_request = false;
        for _ in 0..200 {
            let events = services
                .event_log
                .scan(Some(session.clone()), None, None)
                .await
                .unwrap();
            if events.iter().any(|e| {
                e.kind == "approval.requested"
                    && e.payload.get("run_id").and_then(|v| v.as_str()) == Some(run_id.as_str())
            }) {
                saw_request = true;
                break;
            }
 assert!( !target.exists(), "effect must not run from a refused blanket approve" );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(saw_request, "effectful SuggestOnly turn must surface approval.requested");
        for _ in 0..30 {
 assert!( !target.exists(), "W5: blanket approve must never auto-run a later effectful step" );
            if turn.is_finished() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
 assert!( !target.exists(), "W5: effect must not run without a step-scoped approve" );
        let step_id = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap()
            .into_iter()
            .find(|e| e.kind == "approval.requested")
            .and_then(|e| {
                e.payload
                    .get("step_id")
                    .and_then(|v| v.as_str())
                    .map(StepId::from)
            })
            .expect("step_id on approval.requested");
        let _ = host
            .handle_intent(Intent::Custom {
                name: "deny_effect".to_string(),
                payload: json!({
                    "run_id": run_id.as_str(),
                    "step_id": step_id.as_str(),
                }),
            })
            .await;
        let _ = turn.await;
        assert!(!target.exists(), "deny path leaves the effect unapplied");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn fork_session_from_event_records_ancestry_and_keeps_source_independent() {
        let dir = std::env::temp_dir().join(format!("hide_fork_ancestry_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let source = host.services.session();
        let log = &host.services.event_log;
        log.append(NewEvent::system(
            source.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "one" } }),
        ))
        .await
        .unwrap();
        let boundary = log
            .append(NewEvent::system(
                source.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "two" }),
            ))
            .await
            .unwrap();
        log.append(NewEvent::system(
            source.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "three" }),
        ))
        .await
        .unwrap();
        let (fork_id, record, projection) = host
            .fork_session_from_event(source.clone(), Some(&boundary.id))
            .await
            .unwrap();
        assert_ne!(fork_id, source, "the fork gets a fresh session id");
        assert_eq!(projection.session_id, fork_id);
        let fork_events = log.scan(Some(fork_id.clone()), None, None).await.unwrap();
        assert_eq!(fork_events.len(), 2, "fork = source prefix up to the boundary");
        assert!(!fork_events .iter() .any(|e| e.payload.get("text").and_then(|t| t.as_str()) == Some("three")));
        assert_eq!(record.parent_session_id.as_ref(), Some(&source));
        assert_eq!(record.forked_at, Some(boundary.seq));
        assert_eq!(record.forked_at_event.as_ref(), Some(&boundary.id));
        assert_eq!(record.origin, "fork");
        let looked_up = host
            .services
            .sessions
            .session_record(&host.services.key_value_store, &fork_id)
            .expect("ancestry is durably recorded");
        assert_eq!(looked_up, record, "the KV record matches the returned one");
 assert_eq!( log.scan(Some(source.clone()), None, None).await.unwrap().len(), 3 );
        log.append(NewEvent::system(
            fork_id.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "fork-only" }),
        ))
        .await
        .unwrap();
        assert_eq!(log.scan(Some(source), None, None).await.unwrap().len(), 3);
        assert_eq!(log.scan(Some(fork_id), None, None).await.unwrap().len(), 3);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn fork_session_intent_forks_and_surfaces_new_thread() {
        let dir = std::env::temp_dir().join(format!("hide_fork_intent_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let source = host.services.session();
        let log = &host.services.event_log;
        log.append(NewEvent::system(
            source.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "alpha" } }),
        ))
        .await
        .unwrap();
        let boundary = log
            .append(NewEvent::system(
                source.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "beta" }),
            ))
            .await
            .unwrap();
        let mut rx = host.subscribe_ui();
        let ack = host
            .handle_intent(Intent::ForkSession {
                session_id: source.clone(),
                at_event: boundary.id.clone(),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "fork_session intent is recorded + accepted");
        let ev = loop {
            let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
                .await
                .expect("a UiEvent should arrive")
                .expect("broadcast delivers");
            if let UiEventKind::Custom(ref v) = ev.kind {
                if v.get("kind").and_then(|k| k.as_str()) == Some("session_forked") {
                    break ev;
                }
            }
        };
        let new_id = ev.session_id.clone().expect("fork carries a new session id");
        assert_ne!(new_id, source, "the surfaced thread is a fresh session");
        let record = host
            .services
            .sessions
            .session_record(&host.services.key_value_store, &new_id)
            .expect("the intent path durably records ancestry");
        assert_eq!(record.parent_session_id.as_ref(), Some(&source));
        assert_eq!(record.forked_at, Some(boundary.seq));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_search_transcript_scopes_and_finds_across_sessions() {
        let dir = std::env::temp_dir().join(format!("hide_host_search_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let a = host.services.session();
        let b = host.services.session_named("second");
        assert_ne!(a, b);
        let log = &host.services.event_log;
        log.append(NewEvent::system(
            a.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "fix the ZZALPHA bug" } }),
        ))
        .await
        .unwrap();
        log.append(NewEvent::system(
            b.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "ship ZZBETA feature" } }),
        ))
        .await
        .unwrap();
        let hits = host
            .search_transcript(&crate::replay::TranscriptQuery::literal("ZZALPHA"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "only the ZZALPHA item matches");
        assert_eq!(hits[0].session_id, a);
        assert_eq!(hits[0].role.as_deref(), Some("user"));
        assert!(hits[0].snippet.contains("ZZALPHA"));
        let b_hits = host
            .search_transcript(
                &crate::replay::TranscriptQuery::literal("ZZBETA").in_session(b.clone()),
            )
            .await
            .unwrap();
        assert_eq!(b_hits.len(), 1);
        assert_eq!(b_hits[0].session_id, b);
        let _ = std::fs::remove_dir_all(dir);
    }
    async fn seed_parent_with_boundary(
        log: &hide_core::persistence::DynEventLog,
        parent: &SessionId,
    ) -> Event {
        log.append(NewEvent::system(
            parent.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "explore option A" } }),
        ))
        .await
        .unwrap();
        let boundary = log
            .append(NewEvent::system(
                parent.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "here is option A" }),
            ))
            .await
            .unwrap();
        log.append(NewEvent::system(
            parent.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "post-boundary chatter" }),
        ))
        .await
        .unwrap();
        boundary
    }
    #[tokio::test]
    async fn create_side_chat_is_read_only_inherits_history_and_leaves_parent_independent() {
        let dir = std::env::temp_dir().join(format!("hide_side_chat_create_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let parent = host.services.session();
        let log = &host.services.event_log;
        let boundary = seed_parent_with_boundary(log, &parent).await;
        let (side_id, record, projection) = host
            .create_side_chat(parent.clone(), Some(&boundary.id), true)
            .await
            .unwrap();
        assert_ne!(side_id, parent, "the side chat gets a fresh session id");
        assert_eq!(projection.session_id, side_id);
 assert_eq!( record.relationship, crate::services::SessionRelationship::SideChat );
        assert_eq!(record.origin, "side_chat");
        assert!(record.read_only, "a side chat defaults read-only");
        assert_eq!(record.parent_session_id.as_ref(), Some(&parent));
        assert_eq!(record.forked_at, Some(boundary.seq));
        let looked_up = host
            .services
            .sessions
            .session_record(&host.services.key_value_store, &side_id)
            .expect("side-chat ancestry is durably recorded");
        assert_eq!(looked_up, record, "the KV record matches the returned one");
        let side_events = log.scan(Some(side_id.clone()), None, None).await.unwrap();
 assert_eq!( side_events.len(), 2, "side chat = parent prefix up to the boundary" );
        assert!(!side_events .iter() .any(|e| e.payload.get("text").and_then(|t| t.as_str()) == Some("post-boundary chatter")));
 assert_eq!( log.scan(Some(parent.clone()), None, None).await.unwrap().len(), 3 );
        log.append(NewEvent::system(
            side_id.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "side-only note" }),
        ))
        .await
        .unwrap();
        assert_eq!(log.scan(Some(parent), None, None).await.unwrap().len(), 3);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn merge_side_chat_summary_lands_on_parent_and_side_chat_stays_intact() {
        let dir = std::env::temp_dir().join(format!("hide_side_chat_merge_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let parent = host.services.session();
        let log = &host.services.event_log;
        let boundary = seed_parent_with_boundary(log, &parent).await;
        let (side_id, _record, _projection) = host
            .create_side_chat(parent.clone(), Some(&boundary.id), true)
            .await
            .unwrap();
        let side_before = log.scan(Some(side_id.clone()), None, None).await.unwrap().len();
        let parent_before = log.scan(Some(parent.clone()), None, None).await.unwrap().len();
        log.append(NewEvent::system(
            side_id.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "explored the ZZSIDE alternative" }),
        ))
        .await
        .unwrap();
        let summary = "ZZMERGE: option A is viable; the build verifies";
        let merged = host
            .merge_side_chat_summary(side_id.clone(), parent.clone(), summary)
            .await
            .unwrap();
        assert_eq!(merged.session_id, parent, "the merge event lands on the parent");
        assert_eq!(merged.kind, "session.merge_summary");
 assert_eq!( merged.payload.get("side_chat").and_then(|v| v.as_str()), Some(side_id.as_str()) );
 assert_eq!( merged.payload.get("summary").and_then(|v| v.as_str()), Some(summary) );
        let parent_events = log.scan(Some(parent.clone()), None, None).await.unwrap();
        assert_eq!(parent_events.len(), parent_before + 1);
        assert!(parent_events.iter().any(|e| e.kind == "session.merge_summary"));
        let side_events = log.scan(Some(side_id.clone()), None, None).await.unwrap();
        assert!(!side_events.iter().any(|e| e.kind == "session.merge_summary"));
        assert_eq!(side_events.len(), side_before + 1);
        let hits = host
            .search_transcript(
                &crate::replay::TranscriptQuery::literal("ZZMERGE").in_session(parent.clone()),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "the merged summary is searchable on the parent");
        assert_eq!(hits[0].session_id, parent);
        assert_eq!(hits[0].kind, "session.merge_summary");
        assert_eq!(hits[0].role.as_deref(), Some("side_chat"));
        assert!(hits[0].snippet.contains("ZZMERGE"));
        let side_hits = host
            .search_transcript(
                &crate::replay::TranscriptQuery::literal("ZZSIDE").in_session(side_id.clone()),
            )
            .await
            .unwrap();
        assert_eq!(side_hits.len(), 1, "the side chat's content is intact");
        assert_eq!(side_hits[0].session_id, side_id);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn discarding_a_side_chat_leaves_the_parent_event_count_unchanged() {
        let dir = std::env::temp_dir().join(format!("hide_side_chat_discard_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let parent = host.services.session();
        let log = &host.services.event_log;
        let boundary = seed_parent_with_boundary(log, &parent).await;
        let parent_before = log.scan(Some(parent.clone()), None, None).await.unwrap().len();
        let (side_id, _record, _projection) = host
            .create_side_chat(parent.clone(), Some(&boundary.id), true)
            .await
            .unwrap();
        log.append(NewEvent::system(
            side_id,
            "agent.message",
            json!({ "role": "assistant", "text": "discarded exploration" }),
        ))
        .await
        .unwrap();
        let parent_after = log.scan(Some(parent), None, None).await.unwrap();
        assert_eq!(parent_after.len(), parent_before);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn conversation_graph_tags_children_by_relationship_and_walks_ancestry() {
        use crate::services::{SessionRecord, SessionRelationship};
        let dir = std::env::temp_dir().join(format!("hide_conv_graph_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let parent = host.services.session();
        let log = &host.services.event_log;
        let boundary = seed_parent_with_boundary(log, &parent).await;
        let (fork_id, _r, _p) = host
            .fork_session_from_event(parent.clone(), Some(&boundary.id))
            .await
            .unwrap();
        let (side_id, _r, _p) = host
            .create_side_chat(parent.clone(), Some(&boundary.id), true)
            .await
            .unwrap();
        let ephemeral_id = SessionId::new();
        let ephemeral_rec = SessionRecord::ephemeral_fork(
            ephemeral_id.clone(),
            parent.clone(),
            boundary.seq,
            Some(boundary.id.clone()),
        );
        host.services
            .sessions
            .record_session(&host.services.key_value_store, &ephemeral_rec);
        let graph = host.conversation_graph(&parent);
        assert_eq!(graph.node.session_id, parent);
        assert_eq!(graph.node.relationship, SessionRelationship::Root);
        assert!(graph.node.parent_session_id.is_none());
        assert_eq!(graph.children.len(), 3, "parent has three direct children");
        let child = |id: &SessionId| {
            graph
                .children
                .iter()
                .find(|n| &n.session_id == id)
                .unwrap_or_else(|| panic!("child {id} missing from graph"))
        };
        assert_eq!(child(&fork_id).relationship, SessionRelationship::Fork);
        assert_eq!(child(&side_id).relationship, SessionRelationship::SideChat);
        assert!(child(&side_id).read_only, "the side-chat child is read-only");
 assert_eq!( child(&ephemeral_id).relationship, SessionRelationship::EphemeralFork );
        assert_eq!(graph.edges.len(), 3);
        assert!(graph.edges.iter().all(|e| e.parent == parent));
        let edge_children: std::collections::HashSet<_> =
            graph.edges.iter().map(|e| e.child.clone()).collect();
        assert!(
            edge_children.contains(&fork_id)
                && edge_children.contains(&side_id)
                && edge_children.contains(&ephemeral_id)
        );
        assert!(graph.children.windows(2).all(|w| (w[0].created_ms, &w[0].session_id) <= (w[1].created_ms, &w[1].session_id)));
        assert_eq!(graph, host.conversation_graph(&parent));
        let fork_graph = host.conversation_graph(&fork_id);
        assert_eq!(fork_graph.node.session_id, fork_id);
        assert_eq!(fork_graph.node.relationship, SessionRelationship::Fork);
        assert_eq!(fork_graph.node.parent_session_id.as_ref(), Some(&parent));
        assert_eq!(fork_graph.ancestry.len(), 1, "one ancestor: the root parent");
        assert_eq!(fork_graph.ancestry[0].session_id, parent);
 assert_eq!( fork_graph.ancestry[0].relationship, SessionRelationship::Root );
        let _ = std::fs::remove_dir_all(dir);
    }
    fn verify_result_event(session: &SessionId, oracle: &str, pass: bool) -> NewEvent {
        use hide_kernel::verify::oracle::{OracleClass, Verdict};
        let verdict = if pass {
            Verdict::pass(oracle, OracleClass::Deterministic, "all green")
        } else {
            Verdict::fail(
                oracle,
                OracleClass::Deterministic,
                "2 tests failed",
                Vec::new(),
            )
        };
        NewEvent::system(
            session.clone(),
            "verify.result",
            serde_json::to_value(&verdict).unwrap(),
        )
    }
    #[tokio::test]
    async fn goal_set_evaluate_is_deterministic_over_verify_result_evidence() {
        let dir = std::env::temp_dir().join(format!("hide_goal_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let record = host
            .goal_set(session.clone(), "tests_pass", vec!["tests".to_string()])
            .unwrap();
        assert_eq!(record.status, GoalStatus::Active);
        let got = host.goal_get(&session).expect("goal is durably stored");
        assert_eq!(got, record);
        let v0 = host.goal_evaluate(&session).await.unwrap();
        assert_eq!(v0.outcome, GoalOutcome::NotMet);
 assert!( v0.reason.contains("tests"), "reason names the oracle: {}", v0.reason );
        let log = &host.services.event_log;
        log.append(verify_result_event(&session, "tests", false))
            .await
            .unwrap();
        let vf = host.goal_evaluate(&session).await.unwrap();
        assert_eq!(vf.outcome, GoalOutcome::NotMet);
        assert!(vf.reason.to_lowercase().contains("did not pass"));
        assert_eq!(vf.evidence.len(), 1, "the consulted fail verdict is evidence");
        assert_eq!(host.goal_get(&session).unwrap().status, GoalStatus::Active);
        log.append(verify_result_event(&session, "tests", true))
            .await
            .unwrap();
        let vp = host.goal_evaluate(&session).await.unwrap();
        assert_eq!(vp.outcome, GoalOutcome::Met);
        assert!(vp.is_met());
        assert_eq!(vp.evidence.len(), 1);
        assert_eq!(host.goal_get(&session).unwrap().status, GoalStatus::Met);
        let cleared = host.goal_clear(&session).unwrap().expect("a goal was set");
        assert_eq!(cleared.status, GoalStatus::Cleared);
        assert_eq!(host.goal_get(&session).unwrap().status, GoalStatus::Cleared);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn goal_natural_language_condition_is_deferred_no_model_called() {
        let dir = std::env::temp_dir().join(format!("hide_goal_defer_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        host.goal_set(session.clone(), "the UI feels delightful", Vec::new())
            .unwrap();
        let v = host.goal_evaluate(&session).await.unwrap();
        assert_eq!(v.outcome, GoalOutcome::DeferredModelRequired);
        assert!(v.evidence.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn job_create_persists_and_survives_a_fresh_host_restart() {
        use crate::services::{Budget, JobRecord, Schedule, Trigger};
        let dir = std::env::temp_dir().join(format!("hide_job_recover_{}", now_ms()));
        let session;
        let job_id;
        {
            let host = BackendHost::open_workspace(&dir).unwrap();
            session = host.services.session();
            let budget = Budget {
                max_wall_secs: Some(600),
                max_steps: Some(40),
                ..Budget::default()
            };
            let job = JobRecord::pending(
                session.clone(),
                vec![
                    Trigger::FileChange("src/**/*.rs".to_string()),
                    Trigger::GitPush,
                ],
                budget,
            )
            .with_goal("goal_abc")
            .with_repo("repo_main")
            .with_schedule(Schedule::new("0 9 * * 1-5").with_timezone("UTC"));
            let created = host.job_create(job).await.unwrap();
            job_id = created.job_id.clone();
            assert_eq!(created.status, JobStatus::Pending);
            let events = host
                .services
                .event_log
                .scan(Some(session.clone()), None, None)
                .await
                .unwrap();
            assert!(events.iter().any(|e| e.kind == "job.created"));
        }
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        let recovered = reopened.jobs_recover();
        assert_eq!(recovered.len(), 1, "the pending job survives restart");
        assert_eq!(recovered[0].job_id, job_id);
        assert_eq!(recovered[0].status, JobStatus::Pending);
        assert_eq!(recovered[0].goal_id.as_deref(), Some("goal_abc"));
        assert_eq!(recovered[0].repo_id.as_deref(), Some("repo_main"));
        let got = reopened.job_get(&job_id).expect("job is durably stored");
        assert_eq!(got.triggers.len(), 2);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn job_evaluate_triggers_matches_glob_and_manual_deterministically() {
        use crate::services::{Budget, JobRecord, Trigger, TriggerEvent};
        let dir = std::env::temp_dir().join(format!("hide_job_triggers_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let job = JobRecord::pending(
            session.clone(),
            vec![
                Trigger::FileChange("src/**/*.rs".to_string()),
                Trigger::Manual,
            ],
            Budget::default(),
        );
        assert!(host.job_evaluate_triggers( &job, &TriggerEvent::FileChange("src/host/mod.rs".to_string()), ));
        assert!(!host.job_evaluate_triggers( &job, &TriggerEvent::FileChange("docs/readme.md".to_string()), ));
        assert!(host.job_evaluate_triggers(&job, &TriggerEvent::Manual));
        assert!(!host.job_evaluate_triggers(&job, &TriggerEvent::GitPush));
        let manual_only = JobRecord::pending(
            session.clone(),
            vec![Trigger::Manual],
            Budget::default(),
        );
        assert!(host.job_evaluate_triggers(&manual_only, &TriggerEvent::Manual));
        assert!(!host.job_evaluate_triggers( &manual_only, &TriggerEvent::FileChange("src/lib.rs".to_string()), ));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn job_status_transitions_and_cancel_are_durable_and_recovery_excludes_terminal() {
        use crate::services::{Budget, JobRecord, Trigger};
        let dir = std::env::temp_dir().join(format!("hide_job_status_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let a = host
            .job_create(JobRecord::pending(
                session.clone(),
                vec![Trigger::Manual],
                Budget::default(),
            ))
            .await
            .unwrap();
        let b = host
            .job_create(JobRecord::pending(
                session.clone(),
                vec![Trigger::CiFailure],
                Budget::default(),
            ))
            .await
            .unwrap();
        let c = host
            .job_create(JobRecord::pending(
                session.clone(),
                vec![Trigger::GitPush],
                Budget::default(),
            ))
            .await
            .unwrap();
        let running = host
            .job_update_status(&a.job_id, JobStatus::Running, None)
            .await
            .unwrap()
            .expect("job A exists");
        assert_eq!(running.status, JobStatus::Running);
        assert!(running.updated_ms >= a.created_ms);
        host.job_update_status(&a.job_id, JobStatus::Done, None)
            .await
            .unwrap();
        let blocked = host
            .job_update_status(
                &c.job_id,
                JobStatus::Blocked,
                Some("waiting on upstream push".to_string()),
            )
            .await
            .unwrap()
            .expect("job C exists");
        assert_eq!(blocked.status, JobStatus::Blocked);
        assert_eq!(blocked.last_error.as_deref(), Some("waiting on upstream push"));
        let cancelled = host.job_cancel(&b.job_id).await.unwrap().expect("job B exists");
        assert_eq!(cancelled.status, JobStatus::Cancelled);
        assert!(host
            .job_update_status("job_missing", JobStatus::Running, None)
            .await
            .unwrap()
            .is_none());
        assert!(host.job_cancel("job_missing").await.unwrap().is_none());
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        let recovered = reopened.jobs_recover();
        assert_eq!(recovered.len(), 1, "only the Blocked job is active");
        assert_eq!(recovered[0].job_id, c.job_id);
        assert_eq!(recovered[0].status, JobStatus::Blocked);
 assert_eq!( reopened.job_get(&a.job_id).unwrap().status, JobStatus::Done );
 assert_eq!( reopened.job_get(&b.job_id).unwrap().status, JobStatus::Cancelled );
        assert_eq!(reopened.job_list().len(), 3);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn checkpoint_create_and_restore_folds_source_and_verifies_integrity() {
        let dir = std::env::temp_dir().join(format!("hide_ckpt_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let log = &host.services.event_log;
        log.append(NewEvent::system(
            session.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "one" } }),
        ))
        .await
        .unwrap();
        let boundary = log
            .append(NewEvent::system(
                session.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "two" }),
            ))
            .await
            .unwrap();
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "three" }),
        ))
        .await
        .unwrap();
        let ckpt = host
            .checkpoint_create(session.clone(), Some(&boundary.id), "before-three")
            .await
            .unwrap();
        assert_eq!(ckpt.at_seq, boundary.seq, "the boundary seq is pinned");
        assert_eq!(ckpt.at_event.as_ref(), Some(&boundary.id));
        assert!(ckpt.verify_integrity(), "the sealed integrity digest verifies");
        let list = host.checkpoint_list(&session);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].checkpoint_id, ckpt.checkpoint_id);
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "four" }),
        ))
        .await
        .unwrap();
 assert_eq!( log.scan(Some(session.clone()), None, None).await.unwrap().len(), 5 );
        assert!(
            log.scan(Some(session.clone()), None, None)
                .await
                .unwrap()
                .iter()
                .any(|e| e.kind == "checkpoint.created"
                    && e.payload.get("checkpoint_id").and_then(|v| v.as_str())
                        == Some(ckpt.checkpoint_id.as_str())),
            "the sealed checkpoint is recorded durably, not only on the live bus"
        );
        let (restored, ancestry, projection) =
            crate::tools::with_approved_writes(host.checkpoint_restore(&ckpt.checkpoint_id))
                .await
                .unwrap();
        assert_ne!(restored, session, "restore mints a fresh session");
        assert_eq!(projection.session_id, restored);
        let restored_events = log.scan(Some(restored.clone()), None, None).await.unwrap();
 assert_eq!( restored_events.len(), 2, "restored = source folded to the checkpoint boundary" );
        assert!(!restored_events
            .iter()
            .any(|e| e.payload.get("text").and_then(|t| t.as_str()) == Some("three")));
        assert!(!restored_events
            .iter()
            .any(|e| e.payload.get("text").and_then(|t| t.as_str()) == Some("four")));
        assert_eq!(ancestry.parent_session_id.as_ref(), Some(&session));
        assert_eq!(ancestry.forked_at, Some(boundary.seq));
        assert_eq!(ancestry.forked_at_event.as_ref(), Some(&boundary.id));
        let looked_up = host
            .services
            .sessions
            .session_record(&host.services.key_value_store, &restored)
            .expect("restore records ancestry durably");
        assert_eq!(looked_up, ancestry);
        assert_eq!(log.scan(Some(session.clone()), None, None).await.unwrap().len(), 5);
        assert!(
            crate::tools::with_approved_writes(host.checkpoint_restore("ckpt_does-not-exist"))
                .await
                .is_err()
        );
        let mut tampered = ckpt.clone();
        tampered.at_seq = boundary.seq + 5;
        CheckpointStore::put(&host.services.key_value_store, &tampered).unwrap();
        let err = crate::tools::with_approved_writes(host.checkpoint_restore(&ckpt.checkpoint_id))
            .await
            .unwrap_err();
 assert!( err.to_string().to_lowercase().contains("integrity"), "the tamper is caught: {err}" );
        let _ = std::fs::remove_dir_all(dir);
    }
