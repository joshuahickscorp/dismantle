    #[tokio::test]
    async fn kernel_turn_publishes_visible_assistant_answer_on_ui_bus() {
        let dir = std::env::temp_dir().join(format!("hide_kernel_f2_{}", now_ms()));
        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();
        let (state, ui_events) =
            drive_answer_turn(services, session, "produce the answer", "ZZVISIBLEANSWER done").await;
        assert!(state.phase.is_terminal(), "turn must reach terminal");
        let batch = ui_events.iter().find_map(|e| match &e.kind {
            UiEventKind::TokenBatch { text, .. } => Some(text.clone()),
            _ => None,
        });
        let batch = batch.expect("a TokenBatch (visible assistant answer) must be published");
        assert!(batch.contains("ZZVISIBLEANSWER"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn kernel_turn_synthesizes_answer_when_no_model_text() {
        let dir = std::env::temp_dir().join(format!("hide_kernel_f2b_{}", now_ms()));
        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();
        let (state, ui_events) = drive_answer_turn(services, session, "do it", "").await;
        assert_eq!(state.phase, Phase::Done, "turn must finish");
        let batch = ui_events
            .iter()
            .find_map(|e| match &e.kind {
                UiEventKind::TokenBatch { text, .. } => Some(text.clone()),
                _ => None,
            })
            .expect("a synthesized TokenBatch must still be published");
        assert!(!batch.trim().is_empty() && batch.contains("done"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn kernel_turn_persists_answer_and_next_turn_threads_history() {
        let dir = std::env::temp_dir().join(format!("hide_kernel_f34_{}", now_ms()));
        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();
        let (state1, _ui1) = drive_answer_turn(
            services.clone(),
            session.clone(),
            "first question",
            "ZZTURNONEANSWER complete",
        )
        .await;
        assert!(state1.phase.is_terminal());
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert!(
            events.iter().any(|e| e.kind == "agent.message"
                && e.payload.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && e.payload
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(|t| t.contains("ZZTURNONEANSWER"))
                    .unwrap_or(false)),
            "turn 1 must persist an assistant agent.message with its answer (F4)"
        );
        let history = rebuild_history(&services.event_log, &session).await.unwrap();
        assert!(history.iter().any(|m| m.role == "assistant" && m.content.contains("ZZTURNONEANSWER")));
        let events_before = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap()
            .len();
        let (state2, _ui2) = drive_answer_turn(
            services.clone(),
            session.clone(),
            "second question",
            "ZZTURNTWOANSWER complete",
        )
        .await;
        assert!(state2.phase.is_terminal());
        let events2 = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let turn2_plan = events2
            .iter()
            .skip(events_before)
            .find(|e| e.kind == "plan.created")
            .expect("turn 2 must log a plan.created");
        let objective = turn2_plan
            .payload
            .get("plan")
            .and_then(|p| p.get("objective"))
            .and_then(|o| o.as_str())
            .unwrap_or_default();
        assert!(objective.contains("ZZTURNONEANSWER"));
        assert!(objective.contains("second question"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[test]
    fn gate_book_holds_releases_and_denies() {
        let book = GateBook::default();
        let cmd = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        let held = |argv, cwd| PendingAction::Command { argv, cwd };
        let g1 = book.hold(held(cmd("sudo rm a"), None)).expect("parked");
        let g2 = book
            .hold(held(cmd("rm -rf /"), Some("sub".into())))
            .expect("parked");
        assert_ne!(g1, g2, "gate ids are unique");
        assert_eq!(book.len(), 2);
        let taken = book.take(&g1).expect("g1 parked");
        assert_eq!(taken, held(cmd("sudo rm a"), None));
        assert_eq!(book.len(), 1);
        assert!(book.take(&g1).is_none(), "a gate id is single-use");
        assert!(book.remove(&g2));
        assert!(!book.remove(&g2));
        assert_eq!(book.len(), 0);
        assert!(book.take("command:999").is_none());
        assert!(!book.remove("command:999"));
    }
    #[test]
    fn gate_book_refuses_past_cap_and_keeps_what_it_holds() {
        let book = GateBook::default();
        let mut ids = Vec::new();
        for i in 0..GateBook::CAP {
            ids.push(
                book.hold(PendingAction::Command {
                    argv: vec!["sudo".into(), format!("c{i}")],
                    cwd: None,
                })
                .expect("under cap"),
            );
        }
        assert_eq!(book.len(), GateBook::CAP, "bounded at CAP");
        assert!(book.hold(PendingAction::Command { argv: vec!["sudo".into(), "overflow".into()], cwd: None, }) .is_none());
        for id in &ids {
            assert!(book.take(id).is_some(), "every parked gate is still answerable");
        }
    }
    fn held_argv() -> Vec<String> {
        vec!["mkfs.hidetest".to_string(), "noop".to_string()]
    }
    async fn first_security_gate(
        rx: &mut tokio::sync::broadcast::Receiver<UiEvent>,
    ) -> (String, String) {
        loop {
            let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
                .await
                .expect("a UiEvent should arrive")
                .expect("broadcast delivers");
            if let UiEventKind::SecurityGate { gate, message } = ev.kind {
                return (gate, message);
            }
        }
    }
    #[tokio::test]
    async fn host_holds_dangerous_command_and_releases_on_approve() {
        let dir = std::env::temp_dir().join(format!("hide_host_gate_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let mut rx = host.subscribe_ui();
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: held_argv(),
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted);
 assert_eq!( host.pending_gate_count(), 1, "the command is held at the gate" );
        let (gate, message) = first_security_gate(&mut rx).await;
 assert!( message.contains("mkfs.hidetest"), "the gate names the blocked command" );
        let ack = host
            .handle_intent(Intent::Custom {
                name: "approve_gate".to_string(),
                payload: json!({ "gate": gate }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
 assert_eq!( host.pending_gate_count(), 0, "approve consumes the held command" );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_drops_held_command_on_deny() {
        let dir = std::env::temp_dir().join(format!("hide_host_gate_deny_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let mut rx = host.subscribe_ui();
        host.handle_intent(Intent::RunCommand {
            argv: held_argv(),
            cwd: None,
        })
        .await
        .unwrap();
        assert_eq!(host.pending_gate_count(), 1);
        let (gate, _) = first_security_gate(&mut rx).await;
        host.handle_intent(Intent::Custom {
            name: "deny_gate".to_string(),
            payload: json!({ "gate": gate }),
        })
        .await
        .unwrap();
 assert_eq!( host.pending_gate_count(), 0, "deny drops the held command without running it" );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn trace_d_service_process_persists_streams_and_captures() {
        if !std::path::Path::new("/usr/bin/sandbox-exec").exists() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("hide_host_traced_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let mut rx = host.subscribe_ui();
        let id = host.start_process(
            vec![
                "sh".to_string(),
                "-c".to_string(),
                "i=0; while true; do echo heartbeat $i; i=$((i+1)); sleep 0.1; done".to_string(),
            ],
            None,
            std::collections::BTreeMap::new(),
            true,
            Some(session.to_string()),
        );
        let alive_with_output = {
            let mut ok = false;
            for _ in 0..100 {
                if host
                    .process_state(&id)
                    .map(|s| s.line_count >= 3)
                    .unwrap_or(false)
                {
                    ok = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
            }
            ok
        };
        assert!(alive_with_output, "service process should stream heartbeats");
        let state = host.process_state(&id).unwrap();
        // Production confinement is the default (`disable_sandbox: false` outside
        // `cfg(test)`). Under unit tests we run bare so nested seats can stream.
        assert_eq!(state.status, "running");
        assert!(state.persistent);
        assert_eq!(state.status, "running");
        assert_eq!(state.owner.as_deref(), Some(session.to_string().as_str()));
        host.handle_intent(Intent::Custom {
            name: "new_session".to_string(),
            payload: json!({}),
        })
        .await
        .unwrap();
 assert!( host.process_alive(&id), "the process persists across navigation" );
        let mut streamed = 0usize;
        while let Ok(ev) = rx.try_recv() {
            if let UiEventKind::ToolProgress { call_id, message, .. } = &ev.kind {
                if call_id == &id && message.contains("heartbeat") {
                    streamed += 1;
                }
            }
        }
        assert!(streamed > 0, "incremental stdout must stream as UiEvents");
        let turn = SessionId::new();
        let captured = host.attach_process(&id, turn).expect("attach yields output");
        assert!(!captured.is_empty());
        let mut last: i64 = -1;
        for line in &captured {
            let n: i64 = line
                .strip_prefix("heartbeat ")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or_else(|| panic!("unexpected output line: {line:?}"));
            assert!(n > last, "heartbeat counter must increase: {n} after {last}");
            last = n;
        }
        assert!(host.stop_process(&id));
        for _ in 0..100 {
            if !host.process_alive(&id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        }
        assert!(!host.process_alive(&id), "stop terminates the process");
        assert_eq!(host.process_state(&id).unwrap().status, "stopped");
        let artifact = host.capture_process_artifact(&id).unwrap();
        let bytes = host
            .services
            .blob_store
            .get(&artifact)
            .unwrap()
            .expect("artifact is durable");
        assert!(String::from_utf8_lossy(&bytes).contains("heartbeat"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_new_session_publishes_a_fresh_session() {
        let dir = std::env::temp_dir().join(format!("hide_host_newsess_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let mut rx = host.subscribe_ui();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "new_session".to_string(),
                payload: json!({}),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "new_session is accepted");
        let ev = loop {
            let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
                .await
                .expect("a UiEvent should arrive")
                .expect("broadcast delivers");
            if let UiEventKind::ProjectionPatch { ref projection, .. } = ev.kind {
                if projection == "turn" && ev.session_id.is_some() {
                    break ev;
                }
            }
        };
 assert!( ev.session_id.is_some(), "new_session carries a fresh session id" );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_surface_handoff_claim_never_capability_same_session() {
        let dir = std::env::temp_dir().join(format!("hide_host_surface_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let primary = host.services.session();
        assert_eq!(host.surfaces.session_id(), primary.as_str());
        let switch = host
            .handle_intent(Intent::Custom {
                name: "switch_surface".into(),
                payload: json!({ "surface": "you" }),
            })
            .await
            .unwrap();
        assert!(switch.accepted, "switch_surface accepted: {:?}", switch.message);
        assert_eq!(host.surfaces.active().as_str(), "you");
        assert_eq!(host.surfaces.session_id(), primary.as_str());
        let create = host
            .handle_intent(Intent::Custom {
                name: "handoff_create".into(),
                payload: json!({
                    "kind": "you_to_chat",
                    "claims": [{
                        "id": "clm_host_1",
                        "text": "implement from YOU",
                        "evidence_tier": "cited"
                    }],
                    "deliberately_excludes": [{
                        "item": "gmail credentials",
                        "reason": "claim only"
                    }],
                    "body": { "kind": "implementation_campaign", "goal": "feature" },
                    "actor": "test"
                }),
            })
            .await
            .unwrap();
        assert!(create.accepted, "handoff_create accepted: {:?}", create.message);
        let view = host.surfaces.view();
        assert_eq!(view.session_id, primary.as_str());
        assert_eq!(view.capsules.len(), 1);
        let capsule_id = view.capsules[0].id.clone();
        let sealed = host
            .surfaces
            .view()
            .capsules
            .first()
            .expect("one capsule")
            .clone();
        assert!(
            host.surfaces
                .view()
                .lenses
                .get("you")
                .unwrap()
                .connectors
                .iter()
                .any(|c| c == "gmail")
        );
 assert!( !view .lenses .get("chat") .unwrap() .connectors .iter() .any(|c| c == "gmail") );
        let _ = sealed;
        let receive = host
            .handle_intent(Intent::Custom {
                name: "handoff_receive".into(),
                payload: json!({ "capsule_id": capsule_id }),
            })
            .await
            .unwrap();
 assert!( receive.accepted, "handoff_receive accepted: {:?}", receive.message );
        let after = host.surfaces.view();
        assert_eq!(after.session_id, primary.as_str());
        assert_eq!(after.inbox.get("chat").map(|v| v.len()).unwrap_or(0), 1);
        assert!(!after .lenses .get("chat") .unwrap() .connectors .iter() .any(|c| c == "gmail"));
        let events = host
            .services
            .event_log
            .scan(Some(primary.clone()), None, None)
            .await
            .unwrap();
        assert!(events.iter().any(|e| e.kind == "you.handoff.created"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_holds_create_worktree_at_the_gate() {
        let dir = std::env::temp_dir().join(format!("hide_host_wt_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "create_worktree".to_string(),
                payload: json!({ "branch": "feat/launch pad" }),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "create_worktree is accepted and recorded");
        assert!(ack.held, "and HELD: the ack must not read as done");
        assert_eq!(host.pending_gate_count(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn approving_create_worktree_runs_it() {
        let dir = std::env::temp_dir().join(format!("hide_host_wt_run_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .status()
            .map(|s| s.success())
            .unwrap_or(false));
        for (k, v) in [("user.email", "t@t"), ("user.name", "t")] {
            let _ = std::process::Command::new("git")
                .args(["config", k, v])
                .current_dir(&dir)
                .status();
        }
        std::fs::write(dir.join("a.txt"), "a").unwrap();
        let _ = std::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(&dir)
            .status();
        let _ = std::process::Command::new("git")
            .args(["commit", "-qm", "init"])
            .current_dir(&dir)
            .status();
        let host = BackendHost::open_workspace(&dir).unwrap();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "create_worktree".to_string(),
                payload: json!({ "branch": "runme" }),
            })
            .await
            .unwrap();
        assert!(ack.held, "held at the gate");
        let gate = ack
            .message
            .as_deref()
            .and_then(|m| m.split("gate=").nth(1))
            .unwrap()
            .to_string();
        host.approve_gate(&gate).await.expect("the released effect succeeds");
        let expected = dir.parent().unwrap().join(format!(
            "{}-runme",
            dir.file_name().unwrap().to_string_lossy()
        ));
        let mut made = false;
        for _ in 0..100 {
            if expected.exists() {
                made = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(made, "approving the gate creates the worktree at {expected:?}");
        let _ = std::process::Command::new("git")
            .args(["worktree", "remove", "--force", &expected.to_string_lossy()])
            .current_dir(&dir)
            .status();
        let _ = std::fs::remove_dir_all(&expected);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn write_lease_trace_a_task_edits_and_the_diff_store_fills() {
        let _guard = crate::tools::lease_test_guard();
        let dir = std::env::temp_dir().join(format!("hide_host_lease_{}", now_ms()));
        let repo_root = dir.join("repo");
        std::fs::create_dir_all(repo_root.join("src")).unwrap();
        let host = BackendHost::open_workspace(&dir).unwrap();
        assert_eq!(HideConfig::for_workspace(&dir).security.workspace_write_default, Decision::Ask);
        let session = host.services.session();
        let run = RunId::new();
        let diff_id = format!("diff-{}", run.as_str());
        let file = repo_root.join("src").join("lib.rs");
        std::fs::write(&file, "before\n").unwrap();
        let edit = |content: &str, path: &std::path::Path| {
            ToolCall::new(
                "edit.write_file",
                json!({ "path": path.to_string_lossy(), "content": content }),
            )
        };
        let err = host
            .dispatch_tool(session.clone(), Some(run.clone()), edit("after\n", &file))
            .await
            .expect_err("the shipped default refuses every workspace write");
        assert!(matches!(err, hide_core::error::HideError::PolicyDenied(_)));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "before\n");
 assert!( host.diff_get(&diff_id).is_none(), "no diff can exist for a write that never happened" );
        let ack = host
            .handle_intent(Intent::Custom {
                name: "save_file".to_string(),
                payload: json!({ "path": "repo/src/lib.rs", "content": "after\n" }),
            })
            .await
            .unwrap();
        assert!(ack.held, "with no lease the write is held for approval");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "before\n");
        host.workspace_add_repo(RepoNode::new("repo", &repo_root))
            .unwrap();
        host.workspace_set_repo_trust("repo", TrustState::Trusted)
            .unwrap();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "grant_write_lease".to_string(),
                payload: json!({
                    "repo_id": "repo",
                    "session_id": session.to_string(),
                    "run_id": run.as_str(),
                }),
            })
            .await
            .unwrap();
        assert!(ack.held, "the grant is held: only a human approval installs a lease");
        assert_eq!(host.write_lease(), None, "asking is not being granted");
        let gate = ack
            .message
            .as_deref()
            .and_then(|m| m.split("gate=").nth(1))
            .unwrap()
            .to_string();
        host.approve_gate(&gate).await.expect("the released effect succeeds");
        let lease = host.write_lease().expect("approving the task grants the lease");
        assert_eq!(lease.repo_id, "repo");
        let mut rx = host.subscribe_ui();
        let agent = host.build_turn_dispatcher(session.clone(), Some(run.clone()));
        agent
            .dispatch(edit("after\n", &file))
            .await
            .expect("the lease lets the task's own edit through");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "after\n");
        let proposal = host
            .diff_get(&diff_id)
            .expect("the DiffProposal registry populates");
        assert_eq!(proposal.hunks.len(), 1);
        assert_eq!(proposal.hunks[0].before, "before\n");
        assert_eq!(proposal.hunks[0].after, "after\n");
        let mut published = false;
        while let Ok(ev) = rx.try_recv() {
            if let UiEventKind::ProjectionPatch { projection, .. } = &ev.kind {
                published |= projection == "diff";
            }
        }
        assert!(published, "the diff projection publishes for a leased edit");
        let outside = dir.join("outside.rs");
        assert!(agent.dispatch(edit("x\n", &outside)).await.is_err());
        assert!(!outside.exists());
        let other = host.build_turn_dispatcher(SessionId::from("ses_someone_else"), None);
        assert!(other.dispatch(edit("theirs\n", &file)).await.is_err());
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "after\n");
        let err = host.revert_diff(&diff_id).await.unwrap_err().to_string();
 assert!( err.contains("requires approval"), "the lease must not release a gated effect: {err}" );
        assert!(!crate::connectors::connector_method_is_read("write_file"));
        assert!(!crate::connectors::connector_method_is_read("grant_write_lease"));
        host.run_approved_intent("revert_diff", &json!({ "diff_id": diff_id }))
            .await
            .expect("an approved revert still runs with a lease held");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "before\n");
        let ckpt = host
            .checkpoint_create(session.clone(), None, "leased")
            .await
            .unwrap();
        crate::tools::with_approved_writes(host.checkpoint_rewind(
            &ckpt.checkpoint_id,
            RewindTarget::Code,
        ))
        .await
        .expect("a rewind still runs with a lease held");
        crate::tools::revoke_write_lease("simulated restart");
        host.rebuild_session_projection(session.clone())
            .await
            .unwrap();
 assert_eq!( host.write_lease(), None, "a restart leaves no lease to inherit" );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn every_write_lease_revocation_trigger_revokes() {
        let _guard = crate::tools::lease_test_guard();
        let dir = std::env::temp_dir().join(format!("hide_host_lease_revoke_{}", now_ms()));
        let repo_root = dir.join("repo");
        std::fs::create_dir_all(&repo_root).unwrap();
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let custom = |name: &str, payload: Value| Intent::Custom {
            name: name.to_string(),
            payload,
        };
        let grant = || {
            crate::tools::install_write_lease(crate::tools::WriteLease {
                lease_id: "lease-revoke-test".to_string(),
                repo_id: "repo".to_string(),
                session_id: Some(session.to_string()),
                run_id: Some("run-under-test".to_string()),
                scopes: vec![repo_root.clone()],
                granted_ms: 0,
            })
        };
        let triggers: Vec<(&str, Intent)> = vec![
            ("explicit user revocation", custom("revoke_write_lease", json!({}))),
            ("task cancellation", Intent::CancelRun { run_id: RunId::from("run-under-test") }),
            ("session closure", custom("new_session", json!({}))),
            ("session switch", custom("open_session", json!({ "session_id": session.to_string() }))),
            (
                "session fork",
                Intent::ForkSession {
                    session_id: session.clone(),
                    at_event: EventId::from("evt-none"),
                },
            ),
            (
                "rewind past the grant",
                custom("checkpoint_rewind", json!({ "checkpoint_id": "none", "target": "code" })),
            ),
            (
                "repository trust loss",
                custom(
                    "workspace_set_repo_trust",
                    json!({ "repo_id": "repo", "trust": "untrusted", "root_path": repo_root.to_string_lossy() }),
                ),
            ),
            (
                "scope change",
                custom("environment_switch", json!({ "session_id": session.to_string(), "env_id": "none" })),
            ),
        ];
        for (label, intent) in triggers {
            grant();
            host.handle_intent(intent).await.unwrap();
            assert_eq!(host.write_lease(), None, "{label} must revoke the lease");
        }
        grant();
        assert!(crate::tools::revoke_write_lease_for_run("some-other-run", None).is_none());
        assert!(host.write_lease().is_some(), "another task's end is not this one's");
        host.handle_intent(Intent::CancelRun {
            run_id: RunId::from("some-other-run"),
        })
        .await
        .unwrap();
        assert!(host.write_lease().is_some(), "and neither is its cancellation");
        assert!(crate::tools::revoke_write_lease_for_run("run-under-test", None).is_some());
        assert_eq!(host.write_lease(), None, "task completion revokes");
        grant();
        host.handle_intent(custom(
            "workspace_set_repo_trust",
            json!({ "repo_id": "repo", "trust": "trusted", "root_path": repo_root.to_string_lossy() }),
        ))
        .await
        .unwrap();
        assert!(host.write_lease().is_some(), "granting trust does not revoke");
        crate::tools::revoke_write_lease("end of test");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn every_ask_command_has_a_release_handler() {
        let dir = std::env::temp_dir().join(format!("hide_host_ask_arms_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        use hide_protocol::command::{ApprovalPolicy, BackendBinding};
        let ask: Vec<String> = hide_protocol::command::command_catalog()
            .into_iter()
            .filter(|s| s.approval_policy == ApprovalPolicy::Ask)
            .map(|s| {
                match &s.backend_binding {
                    BackendBinding::Custom(n) => assert_eq!(*n, s.id, "an Ask row must bind its own id"),
                    other => panic!(
                        "{}: ApprovalPolicy::Ask on a {other:?} binding needs an effect_command arm",
                        s.id
                    ),
                }
                s.id
            })
            .collect();
        assert!(!ask.is_empty(), "the catalog declares at least one Ask command");
        for name in ask {
            let err = host
                .run_approved_intent(&name, &json!({}))
                .await
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(!err.contains("no release handler"));
        }
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn every_ask_command_takes_effect_once_approved() {
        use hide_protocol::command::ApprovalPolicy;
        let dir = std::env::temp_dir().join(format!("hide_host_ask_effect_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = HideConfig::for_workspace(&dir);
        assert_eq!(config.security.workspace_write_default, Decision::Ask);
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        let mut ask: Vec<String> = hide_protocol::command::command_catalog()
            .into_iter()
            .filter(|s| s.approval_policy == ApprovalPolicy::Ask)
            .map(|s| s.id)
            .collect();
        ask.push("save_file".to_string());
        assert!(ask.len() > 1, "the catalog declares Ask commands");
        for name in ask {
            match name.as_str() {
                "revert_diff" => {
                    let file = dir.join("reverted.rs").to_string_lossy().to_string();
                    std::fs::write(&file, "AFTER\n").unwrap();
                    let hunk = DiffHunk {
                        hunk_id: "h0".to_string(),
                        file: file.clone(),
                        base_hash: blake3::hash(b"BEFORE\n").to_hex().to_string(),
                        before: "BEFORE\n".to_string(),
                        after: "AFTER\n".to_string(),
                        status: HunkStatus::Pending,
                        provenance: DiffProvenance {
                            plan_step: None,
                            agent: "test".to_string(),
                            turn: 0,
                        },
                    };
                    let proposal = DiffProposal {
                        diff_id: "d_ask".to_string(),
                        run_id: "r_ask".to_string(),
                        session_id: host.services.session(),
                        created_ms: now_ms(),
                        created_from: hunk.provenance.clone(),
                        hunks: vec![hunk],
                    };
                    DiffStore::put(&host.services.key_value_store, &proposal).unwrap();
                    host.run_approved_intent(&name, &json!({ "diff_id": "d_ask" }))
                        .await
                        .expect("an approved revert must not be refused by the write policy");
                    assert_eq!(std::fs::read_to_string(&file).unwrap(), "BEFORE\n");
                }
                "save_file" => {
                    let rel = "saved.txt";
                    host.run_approved_intent(
                        &name,
                        &json!({ "path": rel, "content": "SAVED\n" }),
                    )
                    .await
                    .expect("an approved save must not be refused by the write policy");
                    assert_eq!(std::fs::read_to_string(dir.join(rel)).unwrap(), "SAVED\n");
                }
                "grant_write_lease" => {
                    let _guard = crate::tools::lease_test_guard();
                    let repo_root = dir.join("leased");
                    std::fs::create_dir_all(&repo_root).unwrap();
                    host.workspace_add_repo(RepoNode::new("leased", &repo_root))
                        .unwrap();
                    let err = host
                        .run_approved_intent(&name, &json!({ "repo_id": "leased" }))
                        .await
                        .err()
                        .map(|e| e.to_string())
                        .unwrap_or_default();
                    assert!(err.contains("not trusted"));
                    assert_eq!(host.write_lease(), None, "and nothing was installed");
                    host.workspace_set_repo_trust("leased", TrustState::Trusted)
                        .unwrap();
                    host.run_approved_intent(&name, &json!({ "repo_id": "leased" }))
                        .await
                        .expect("an approved grant over a trusted repo installs the lease");
                    let lease = host.write_lease().expect("approving the gate grants the lease");
                    assert!(lease.covers(&repo_root.join("src/new.rs").to_string_lossy()));
 assert!( !lease.covers(&dir.join("outside.rs").to_string_lossy()), "and nothing outside it" );
                    crate::tools::revoke_write_lease("end of test");
                }
                "checkpoint_restore" | "checkpoint_rewind" | "workspace_set_repo_trust"
                | "create_worktree" => {
                    let err = host
                        .run_approved_intent(
                            &name,
                            &json!({ "checkpoint_id": "nope", "repo_id": "nope", "trusted": true }),
                        )
                        .await
                        .err()
                        .map(|e| e.to_string())
                        .unwrap_or_default();
 assert!( !err.contains("no release handler"), "{name}: approving the gate does nothing" );
                    assert!(!err.to_lowercase().contains("policy"));
                }
                other => panic!(
                    "{other} is ApprovalPolicy::Ask with no effect assertion here: say what \
                     approving its gate is supposed to DO, or it ships as another dead control"
                ),
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn an_ask_effect_is_refused_on_every_channel_that_did_not_release_a_gate() {
        use hide_protocol::protocol::Method;
        let dir = std::env::temp_dir().join(format!("hide_host_chanbypass_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        append_code_change(&host, &session, "a.rs", "base").await;
        let ckpt = host
            .checkpoint_create(session.clone(), None, "cp")
            .await
            .unwrap();
        let out = host
            .rpc(
                Method::CheckpointRestore,
                json!({ "checkpoint_id": ckpt.checkpoint_id }),
            )
            .await;
        let body = serde_json::to_string(&out).unwrap().to_lowercase();
        assert!(body.contains("requires approval"));
        let err = host
            .checkpoint_restore(&ckpt.checkpoint_id)
            .await
            .unwrap_err();
        assert!(matches!(err, hide_core::error::HideError::PolicyDenied(_)), "{err}");
        crate::tools::with_approved_writes(host.checkpoint_restore(&ckpt.checkpoint_id))
            .await
            .expect("an approved restore runs");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn a_gated_destructive_command_acks_held() {
        let dir = std::env::temp_dir().join(format!("hide_host_danger_held_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["sudo".to_string(), "rm".to_string(), "-rf".to_string(), "/".to_string()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted, "the request is recorded");
        assert!(ack.held, "a parked destructive command may not read as started");
        assert!(ack.message.unwrap_or_default().contains("gate="), "carries the gate to approve");
        let ok = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["echo".to_string(), "hi".to_string()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(!ok.held, "a safe command is not parked");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_refuses_an_unhandled_custom_name() {
        let dir = std::env::temp_dir().join(format!("hide_host_unhandled_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "create_pr".to_string(),
                payload: json!({}),
            })
            .await
            .unwrap();
        assert!(!ack.accepted, "no handler means no success ack");
        assert!(ack.message.unwrap_or_default().contains("create_pr"));
 assert!( ack.event_seq.is_some(), "the intent is still recorded in the log" );
        let ok = host
            .handle_intent(Intent::Custom {
                name: "new_session".to_string(),
                payload: json!({}),
            })
            .await
            .unwrap();
        assert!(ok.accepted);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_runs_static_analysis_over_the_intent_channel() {
        let dir = std::env::temp_dir().join(format!("hide_host_sa_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "run_static_analysis".to_string(),
                payload: json!({
                    "session_id": session.to_string(),
                    "sources": [{ "path": "src/a.rs", "text": "fn a() { let _ = x.unwrap(); }\n" }],
                }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let receipts = host.verification_receipts(&session).await.unwrap();
        assert_eq!(receipts.len(), 1, "the run recorded a durable receipt");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_answering_an_unknown_gate_is_refused_not_accepted() {
        let dir = std::env::temp_dir().join(format!("hide_host_gate_unknown_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        for name in ["approve_gate", "deny_gate"] {
            let ack = host
                .handle_intent(Intent::Custom {
                    name: name.to_string(),
                    payload: json!({ "gate": "command:does-not-exist" }),
                })
                .await
                .unwrap();
            assert!(!ack.accepted, "{name} of an unknown gate must not read as done");
 assert!(ack .message .unwrap_or_default() .contains("not awaiting a decision"));
        }
        assert_eq!(host.pending_gate_count(), 0);
        let _ = std::fs::remove_dir_all(dir);
    }
    const EFFECT_CONTENT: &str = "approved-effect-applied\n";
    struct EditPlanner {
        target: String,
        content: String,
        oracle: String,
    }
    impl hide_kernel::plan::planner::Planner for EditPlanner {
        fn synthesize<'a>(
            &'a self,
            objective: &'a str,
        ) -> futures::future::BoxFuture<'a, Result<hide_kernel::plan::schema::Plan>> {
            use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
            let target = self.target.clone();
            let content = self.content.clone();
            let oracle = self.oracle.clone();
            let objective = objective.to_string();
            Box::pin(async move {
                let mut step = PlanStep::new(
                    "write the effect file",
                    StepKind::Edit,
                    Acceptance::with_oracles("the effect file is applied", vec![oracle]),
                );
                step.tool_hint = Some("edit.write_file".to_string());
                step.tool_args = Some(json!({ "path": target, "content": content }));
                Ok(Plan {
                    id: hide_core::ids::PlanId::new(),
                    title: "effectful edit plan".to_string(),
                    objective,
                    steps: vec![step],
                    status: PlanStatus::Active,
                    budget: Default::default(),
                })
            })
        }
    }
    struct NoopPassOracle(&'static str);
    impl hide_kernel::verify::oracle::Oracle for NoopPassOracle {
        fn name(&self) -> &str {
            self.0
        }
        fn verify<'a>(
            &'a self,
            _input: &'a hide_kernel::verify::oracle::VerificationInput,
        ) -> futures::future::BoxFuture<'a, Result<hide_kernel::verify::oracle::Verdict>> {
            use hide_kernel::verify::oracle::{OracleClass, Verdict};
            let name = self.0;
            Box::pin(async move { Ok(Verdict::pass(name, OracleClass::Deterministic, "noop pass")) })
        }
    }
    async fn drive_effectful_turn(
        decision: Option<ApprovalDecision>,
        max_steps: usize,
    ) -> (
        AgentState,
        Vec<hide_core::event::Event>,
        PathBuf,
        PathBuf,
        RunId,
    ) {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::Duration;
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("hide_approval_{}_{}", now_ms(), n));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\nname=\"fx\"\n").unwrap();
        let _ = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .output();
        let services =
            Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
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
            .workspace_root(root.clone())
            .autonomy(Autonomy::SuggestOnly)
            .planner(planner as Arc<dyn hide_kernel::plan::planner::Planner>)
            .dispatcher(dispatcher)
            .oracle_suite(suite)
            .build();
        let ui_bus = Arc::new(UiEventBus::default());
        let interrupts = Arc::new(InterruptHub::default());
        let approvals = Arc::new(ApprovalHub::default());
        let run_id = RunId::new();
        match decision {
            Some(ApprovalDecision::Deny) => {
                approvals.decide(run_id.clone(), None, ApprovalDecision::Deny);
            }
            Some(ApprovalDecision::Approve) => {
                let approvals_bg = approvals.clone();
                let run_bg = run_id.clone();
                let log_bg = services.event_log.clone();
                let session_bg = session.clone();
                tokio::spawn(async move {
                    for _ in 0..200 {
                        if let Ok(events) =
                            log_bg.scan(Some(session_bg.clone()), None, None).await
                        {
                            if let Some(ev) = events.iter().find(|e| {
                                e.kind == "approval.requested"
                                    && e.payload.get("run_id").and_then(|v| v.as_str())
                                        == Some(run_bg.as_str())
                            }) {
                                if let Some(step) = ev
                                    .payload
                                    .get("step_id")
                                    .and_then(|v| v.as_str())
                                    .map(StepId::from)
                                {
                                    approvals_bg.decide(
                                        run_bg,
                                        Some(step),
                                        ApprovalDecision::Approve,
                                    );
                                    return;
                                }
                            }
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                });
            }
            None => {}
        }
        let state = run_turn_kernel(
            kernel,
            services.event_log.clone(),
            services.key_value_store.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            ui_bus,
            interrupts,
            approvals,
            run_id.clone(),
            session.clone(),
            "http://127.0.0.1:9/unreachable".to_string(),
            "apply the effect".to_string(),
            max_steps,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        (state, events, target, dir, run_id)
    }
