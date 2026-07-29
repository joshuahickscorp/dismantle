    #[tokio::test]
    async fn goal_and_checkpoint_custom_intents_are_wired() {
        let dir = std::env::temp_dir().join(format!("hide_goal_ckpt_intent_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "goal_set".to_string(),
                payload: json!({
                    "session_id": session.as_str(),
                    "condition": "tests_pass",
                    "acceptance": ["tests"]
                }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let goal = host.goal_get(&session).expect("goal_set intent stored a goal");
        assert_eq!(goal.condition, "tests_pass");
        assert_eq!(goal.acceptance, vec!["tests".to_string()]);
        host.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "hi" }),
            ))
            .await
            .unwrap();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "checkpoint_create".to_string(),
                payload: json!({ "session_id": session.as_str(), "label": "tail" }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let list = host.checkpoint_list(&session);
        assert_eq!(list.len(), 1, "checkpoint_create intent recorded one checkpoint");
        assert_eq!(list[0].label, "tail");
        let _ = std::fs::remove_dir_all(dir);
    }
    async fn append_code_change(
        host: &BackendHost,
        session: &SessionId,
        file: &str,
        after: &str,
    ) -> Event {
        host.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "diff.proposed",
                json!({ "hunks": [ { "file": file, "after": after } ] }),
            ))
            .await
            .unwrap()
    }
    #[tokio::test]
    async fn trace_e_rewind_code_only_then_fork_and_compare() {
        let dir = std::env::temp_dir().join(format!("hide_trace_e_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let log = &host.services.event_log;
        log.append(NewEvent::system(
            session.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "add a feature" } }),
        ))
        .await
        .unwrap();
        append_code_change(&host, &session, "src/a.rs", "fn f() {}").await;
        let ckpt = host
            .checkpoint_create(session.clone(), None, "before-change")
            .await
            .unwrap();
        assert!(ckpt.verify_integrity(), "sealed integrity (boundary + coverage) verifies");
        assert_eq!(ckpt.coverage.repo_state.count, 1, "coverage references the 1 baseline file");
        assert!(ckpt.coverage.live_state_capsule.is_none(), "live capsule stays DEFERRED_MODEL_REQUIRED");
        let base_hash = blake3::hash(b"fn f() {}").to_hex().to_string();
        append_code_change(&host, &session, "src/a.rs", "fn f() { panic!() }").await;
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "explained the change" }),
        ))
        .await
        .unwrap();
        let bad_receipt = log
            .append(NewEvent::system(
                session.clone(),
                "verify.result",
                json!({ "verification_id": "v-1", "scope": ["src/a.rs"], "verdict": { "status": "fail" } }),
            ))
            .await
            .unwrap();
        let rewound = crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Code),
        )
        .await
        .unwrap();
        assert_eq!(rewound.target, RewindTarget::Code);
        let child_code = host.code_state_of(&rewound.session_id, None).await.unwrap();
        assert_eq!(child_code.get("src/a.rs"), Some(&base_hash), "code reverted to the checkpoint");
        let child_events = log.scan(Some(rewound.session_id.clone()), None, None).await.unwrap();
        assert!(
            child_events.iter().any(|e| e.kind == "agent.message"
                && e.payload.get("text").and_then(|t| t.as_str()) == Some("explained the change")),
            "conversation after the boundary is preserved"
        );
        assert!(
            !child_events.iter().any(|e| e.kind == "diff.proposed"
                && e.payload.get("hunks").is_some()
                && e.payload.to_string().contains("panic")),
            "the buggy post-boundary code edit is gone"
        );
        assert_eq!(rewound.reverted_files, vec!["src/a.rs".to_string()]);
        assert!(rewound.invalidated_receipts.contains(&bad_receipt.id));
        let (fp, inherited, own) = rewind::split_inherited_own(&child_events);
        let fp = fp.expect("the rewound child carries a fork.point marker");
        assert_eq!(fp.parent_thread, session, "the marker points at the source thread");
        assert_eq!(fp.start_ordinal, 3, "own history starts after the 2 inherited prefix events");
        assert_eq!(inherited.len(), 2);
        assert!(own.iter().any(|e| e.kind == "agent.message"));
        let alt = host.checkpoint_fork(&ckpt.checkpoint_id).await.unwrap();
        assert_ne!(alt.session_id, rewound.session_id);
        append_code_change(&host, &alt.session_id, "src/a.rs", "fn f() { ok() }").await;
        let comparison = host
            .compare_session_code(&rewound.session_id, &alt.session_id)
            .await
            .unwrap();
        assert_eq!(comparison.files.len(), 1);
        assert_eq!(comparison.files[0].file, "src/a.rs");
        assert_eq!(comparison.files[0].status, rewind::ChangeStatus::Modified);
        let source_code = host.code_state_of(&session, None).await.unwrap();
        assert_eq!(source_code.get("src/a.rs"), Some(&blake3::hash(b"fn f() { panic!() }").to_hex().to_string()));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn checkpoint_rewind_modes_replay_and_inspect() {
        let dir = std::env::temp_dir().join(format!("hide_rewind_modes_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let log = &host.services.event_log;
        append_code_change(&host, &session, "a.rs", "base").await;
        let ckpt = host.checkpoint_create(session.clone(), None, "cp").await.unwrap();
        append_code_change(&host, &session, "a.rs", "edited").await;
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "after" }),
        ))
        .await
        .unwrap();
        log.append(NewEvent::system(
            session.clone(),
            "verify.result",
            json!({ "verification_id": "v", "scope": ["a.rs"], "verdict": { "status": "fail" } }),
        ))
        .await
        .unwrap();
        let both = crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Both),
        )
        .await
        .unwrap();
        let both_events = log.scan(Some(both.session_id.clone()), None, None).await.unwrap();
        let (_, _, both_own) = rewind::split_inherited_own(&both_events);
        assert!(both_own.is_empty(), "both-rewind leaves no post-boundary records");
        assert!(!both.reverted_files.is_empty(), "both reverts the post-boundary code");
        let conv = crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Conversation),
        )
        .await
        .unwrap();
        let conv_code = host.code_state_of(&conv.session_id, None).await.unwrap();
        assert_eq!(conv_code.get("a.rs"), Some(&blake3::hash(b"edited").to_hex().to_string()));
        let conv_events = log.scan(Some(conv.session_id.clone()), None, None).await.unwrap();
        assert!(!conv_events.iter().any(|e| e.kind == "agent.message"));
        assert!(conv.reverted_files.is_empty(), "conversation rewind reverts no code");
        assert!(conv.invalidated_receipts.is_empty(), "no code reverted -> no receipts invalidated");
        let replay = host.checkpoint_replay(&ckpt.checkpoint_id).await.unwrap();
        assert_eq!(replay.replayed_events.len(), 4, "4 post-boundary events replayed");
        let replay_events = log.scan(Some(replay.session_id.clone()), None, None).await.unwrap();
        let (_, _, replay_own) = rewind::split_inherited_own(&replay_events);
        assert_eq!(replay_own.len(), 4, "the replayed events are the child's own history");
        let inspect = host.checkpoint_inspect(&ckpt.checkpoint_id).await.unwrap();
        assert!(inspect.integrity_ok, "sealed integrity verifies");
        assert!(inspect.coverage_current, "coverage matches the untampered log");
        assert!(inspect.drift.is_empty());
        assert_eq!(inspect.invalidated_receipts.len(), 1, "the failing receipt is invalidated by a code rewind");
        let mut tampered = ckpt.clone();
        tampered.coverage.repo_state = StateRef::of(&["a.rs:forged".to_string()]);
        CheckpointStore::put(&host.services.key_value_store, &tampered).unwrap();
        assert!(!host.checkpoint_inspect(&ckpt.checkpoint_id).await.unwrap().integrity_ok);
        assert!(crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Code)
        )
        .await
        .is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn rewind_code_reverts_the_working_tree_and_reports_invalidated_receipts() {
        let dir = std::env::temp_dir().join(format!("hide_rewind_disk_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        let session = host.services.session();
        let log = &host.services.event_log;
        let run = RunId::new();
        let path = dir.join("a.rs").to_string_lossy().to_string();
        std::fs::write(&path, "A0\n").unwrap();
        log.append(NewEvent::system(
            session.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "change a.rs" } }),
        ))
        .await
        .unwrap();
        let ckpt = host
            .checkpoint_create(session.clone(), None, "before")
            .await
            .unwrap();
        let result = host
            .dispatch_tool(
                session.clone(),
                Some(run.clone()),
                ToolCall::new("edit.write_file", json!({ "path": path, "content": "A1\n" })),
            )
            .await
            .unwrap();
        assert_eq!(result.status, ToolStatus::Ok, "the scripted edit applies");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "A1\n");
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "explained the change" }),
        ))
        .await
        .unwrap();
        let receipt = log
            .append(NewEvent::system(
                session.clone(),
                "verify.result",
                json!({ "verification_id": "v-1", "scope": ["a.rs"], "verdict": { "status": "pass" } }),
            ))
            .await
            .unwrap();
        let rewound = crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Code),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "A0\n");
        assert_eq!(rewound.reverted_files, vec!["a.rs".to_string()]);
        assert_eq!(rewound.invalidated_receipts, vec![receipt.id.clone()]);
        let child_events = log
            .scan(Some(rewound.session_id.clone()), None, None)
            .await
            .unwrap();
        assert!(
            child_events.iter().any(|e| e.kind == "agent.message"
                && e.payload.get("text").and_then(|t| t.as_str()) == Some("explained the change")),
            "a code rewind keeps the conversation"
        );
        assert!(child_events .iter() .any(|e| e.kind == "user.intent.submit_turn"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn rewind_intent_refuses_an_omitted_target() {
        let dir = std::env::temp_dir().join(format!("hide_rewind_target_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        append_code_change(&host, &session, "a.rs", "base").await;
        let ckpt = host.checkpoint_create(session.clone(), None, "cp").await.unwrap();
        append_code_change(&host, &session, "a.rs", "edited").await;
        let err = host
            .handle_goal_checkpoint_intent(
                "checkpoint_rewind",
                &json!({ "checkpoint_id": ckpt.checkpoint_id }),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("missing 'target'"));
 assert_eq!( host.checkpoint_list(&session).len(), 1, "the refusal is inert" );
        let err = host
            .handle_goal_checkpoint_intent(
                "checkpoint_rewind",
                &json!({ "checkpoint_id": ckpt.checkpoint_id, "target": "" }),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unknown target"), "a blank label is refused: {err}");
        crate::tools::with_approved_writes(host.handle_goal_checkpoint_intent(
            "checkpoint_rewind",
            &json!({ "checkpoint_id": ckpt.checkpoint_id, "target": "conversation" }),
        ))
        .await
        .expect("an explicit target still runs");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn plan_domain_projection_and_mutations_are_wired() {
        use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
        let dir = std::env::temp_dir().join(format!("hide_plan_domain_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let investigate = PlanStep::new(
            "investigate",
            StepKind::Investigate,
            Acceptance::predicate("root cause found"),
        );
        let edit = PlanStep::new(
            "apply the fix",
            StepKind::Edit,
            Acceptance::predicate("build passes"),
        );
        let a = investigate.id.clone();
        let b = edit.id.clone();
        let plan = Plan {
            id: hide_core::ids::PlanId::new(),
            title: "fix".to_string(),
            objective: "make it pass".to_string(),
            steps: vec![investigate, edit],
            status: PlanStatus::Active,
            budget: Default::default(),
        };
        let mut rx = host.subscribe_ui();
        host.publish_plan(&session, &plan, Autonomy::SuggestOnly).unwrap();
        let patch = match rx.recv().await.unwrap().kind {
            UiEventKind::ProjectionPatch { projection, patch } => {
                assert_eq!(projection, "plan");
                patch
            }
            other => panic!("expected a plan ProjectionPatch, got {other:?}"),
        };
        let steps = patch.get("steps").and_then(|v| v.as_array()).unwrap();
        assert_eq!(steps.len(), 2, "the projection carries the real steps");
        let edit_patch = steps.iter().find(|s| s["id"] == json!(b.as_str())).unwrap();
        assert_eq!(edit_patch["write_blocked"], json!(true));
        assert_eq!(edit_patch["acceptance"], json!("build passes"));
        let inv_patch = steps.iter().find(|s| s["id"] == json!(a.as_str())).unwrap();
        assert_eq!(inv_patch["write_blocked"], json!(false));
        let ack = host
            .handle_intent(Intent::Custom {
                name: "approve_plan".to_string(),
                payload: json!({ "session_id": session.as_str() }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let record = host.plan_get(&session).expect("durable plan persisted");
        assert!(record.approved);
        assert!(record.steps.iter().all(|s| s.approved));
        assert!(record.steps.iter().find(|s| s.id == b.as_str()).unwrap().write_blocked);
        host.handle_intent(Intent::Custom {
            name: "edit_plan_step".to_string(),
            payload: json!({
                "session_id": session.as_str(),
                "step_id": a.as_str(),
                "text": "dig deeper"
            }),
        })
        .await
        .unwrap();
        assert_eq!(host.plan_get(&session).unwrap().steps[0].text, "dig deeper");
        host.handle_intent(Intent::Custom {
            name: "reorder_plan".to_string(),
            payload: json!({
                "session_id": session.as_str(),
                "order": [b.as_str(), a.as_str()]
            }),
        })
        .await
        .unwrap();
        let reordered = host.plan_get(&session).unwrap();
        assert_eq!(reordered.steps[0].id, b.as_str());
        assert_eq!(reordered.steps[1].id, a.as_str());
        let mut plan_patches = 0;
        while let Ok(ev) = rx.try_recv() {
            if let UiEventKind::ProjectionPatch { projection, .. } = ev.kind {
                if projection == "plan" {
                    plan_patches += 1;
                }
            }
        }
 assert!( plan_patches >= 3, "approve + edit + reorder each republish, got {plan_patches}" );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn memory_revalidation_holds_active_while_citations_resolve_then_quarantines() {
        let dir = std::env::temp_dir().join(format!("hide_mem_reval_{}", now_ms()));
        let repo = std::env::temp_dir().join(format!("hide_mem_repo_{}", now_ms()));
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(repo.join("src").join("lib.rs"), "pub fn target_symbol() {}\n").unwrap();
        std::fs::write(repo.join("README.md"), "# fixture\n").unwrap();
        let host = BackendHost::open_workspace(&dir).unwrap();
        let scope = crate::memory::MemoryScope::Repo("fixture".to_string());
        let record = host
            .memory_add(
                crate::memory::MemoryDraft::new(
                    scope.clone(),
                    "lib exports target_symbol",
                    "code_index",
                    "planner",
                )
                .with_citations(vec![
                    "README.md".to_string(),
                    "src/lib.rs#target_symbol".to_string(),
                ]),
            )
            .unwrap();
        assert_eq!(record.status, crate::memory::MemoryStatus::Active);
        let pass = host
            .memory_revalidate(
                crate::memory::RevalidateTarget::record(&record.memory_id),
                &repo,
            )
            .unwrap();
        assert_eq!(pass.len(), 1);
        assert!(pass[0].resolved, "citations resolve: {}", pass[0].reason);
        assert!(!pass[0].quarantined);
        assert_eq!(pass[0].status, crate::memory::MemoryStatus::Active);
        assert_eq!(host.memory_context(&scope).len(), 1);
 assert!( host.memory_get(&record.memory_id).unwrap().last_validated_ms >= record.created_ms );
        std::fs::remove_file(repo.join("src").join("lib.rs")).unwrap();
        let fail = host
            .memory_revalidate(
                crate::memory::RevalidateTarget::record(&record.memory_id),
                &repo,
            )
            .unwrap();
        assert!(!fail[0].resolved);
        assert!(fail[0].quarantined, "a vanished citation quarantines");
        assert_eq!(fail[0].status, crate::memory::MemoryStatus::Quarantined);
        assert!(fail[0].reason.contains("no longer resolve"));
        assert_eq!(fail[0].unresolved, vec!["src/lib.rs#target_symbol".to_string()]);
        assert_eq!(host.memory_get(&record.memory_id).unwrap().status, crate::memory::MemoryStatus::Quarantined);
        assert!(host.memory_context(&scope).is_empty());
        assert!(host
            .memory_revalidate(
                crate::memory::RevalidateTarget::record("mem_does-not-exist"),
                &repo,
            )
            .is_err());
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(repo);
    }
    #[tokio::test]
    async fn memory_supersede_replaces_without_erasing_history() {
        let dir = std::env::temp_dir().join(format!("hide_mem_supersede_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let scope = crate::memory::MemoryScope::Repo("proj".to_string());
        let old = host
            .memory_add(crate::memory::MemoryDraft::new(
                scope.clone(),
                "build uses make",
                "docs",
                "user",
            ))
            .unwrap();
        let (old_after, new) = host
            .memory_supersede(
                &old.memory_id,
                crate::memory::MemoryDraft::new(
                    scope.clone(),
                    "build uses cargo",
                    "docs",
                    "user",
                ),
            )
            .unwrap();
        assert_eq!(old_after.status, crate::memory::MemoryStatus::Superseded);
        assert_eq!(old_after.superseded_by.as_deref(), Some(new.memory_id.as_str()));
        assert_eq!(new.supersedes.as_deref(), Some(old.memory_id.as_str()));
        assert_eq!(new.status, crate::memory::MemoryStatus::Active);
        let reloaded_old = host.memory_get(&old.memory_id).expect("old record kept");
        assert_eq!(reloaded_old.status, crate::memory::MemoryStatus::Superseded);
        assert_eq!(reloaded_old.superseded_by.as_deref(), Some(new.memory_id.as_str()));
        assert_eq!(host.memory_list(&scope).len(), 2);
        let context = host.memory_context(&scope);
        assert_eq!(context.len(), 1);
        assert_eq!(context[0].memory_id, new.memory_id);
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        assert_eq!(reopened.memory_list(&scope).len(), 2);
        assert_eq!(reopened.memory_context(&scope).len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn memory_outcome_governance_raises_on_success_and_quarantines_below_floor() {
        let dir = std::env::temp_dir().join(format!("hide_mem_outcome_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let scope = crate::memory::MemoryScope::User("u".to_string());
        let record = host
            .memory_add(crate::memory::MemoryDraft::new(
                scope.clone(),
                "prefer state over text",
                "conversation",
                "user",
            ))
            .unwrap();
        let start = record.outcome_score;
        host.memory_record_outcome(&record.memory_id, true).unwrap();
        let after_success = host.memory_record_outcome(&record.memory_id, true).unwrap();
        assert!(after_success.outcome_score > start);
        assert_eq!(after_success.use_count, 2);
        assert_eq!(after_success.status, crate::memory::MemoryStatus::Active);
        let mut latest = after_success;
        for _ in 0..3 {
            latest = host.memory_record_outcome(&record.memory_id, false).unwrap();
        }
        assert!(latest.outcome_score < crate::memory::QUARANTINE_FLOOR);
        assert_eq!(latest.status, crate::memory::MemoryStatus::Quarantined);
        assert_eq!(host.memory_get(&record.memory_id).unwrap().status, crate::memory::MemoryStatus::Quarantined);
        assert!(host.memory_context(&scope).is_empty());
        assert!(host.memory_record_outcome("mem_missing", true).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn memory_list_returns_only_the_requested_scope() {
        let dir = std::env::temp_dir().join(format!("hide_mem_scope_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = crate::memory::MemoryScope::Session("s1".to_string());
        let repo = crate::memory::MemoryScope::Repo("r1".to_string());
        let user = crate::memory::MemoryScope::User("u1".to_string());
        host.memory_add(crate::memory::MemoryDraft::new(
            session.clone(),
            "session claim",
            "src",
            "a",
        ))
        .unwrap();
        host.memory_add(crate::memory::MemoryDraft::new(
            repo.clone(),
            "repo claim",
            "src",
            "a",
        ))
        .unwrap();
        host.memory_add(crate::memory::MemoryDraft::new(
            user.clone(),
            "user claim",
            "src",
            "a",
        ))
        .unwrap();
        let session_list = host.memory_list(&session);
        assert_eq!(session_list.len(), 1);
        assert_eq!(session_list[0].claim, "session claim");
        assert!(session_list.iter().all(|r| r.scope == session));
        let repo_list = host.memory_list(&repo);
        assert_eq!(repo_list.len(), 1);
        assert_eq!(repo_list[0].claim, "repo claim");
        let user_list = host.memory_list(&user);
        assert_eq!(user_list.len(), 1);
        assert_eq!(user_list[0].claim, "user claim");
        let repo_s1 = crate::memory::MemoryScope::Repo("s1".to_string());
        assert!(host.memory_list(&repo_s1).is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn workspace_graph_projects_repos_and_typed_edges_deterministically() {
        use crate::services::{RepoNode, WorkspaceEdgeKind};
        let dir = std::env::temp_dir().join(format!("hide_ws_graph_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        host.workspace_add_repo(RepoNode::new("web", dir.join("web")).with_branch("main"))
            .unwrap();
        host.workspace_add_repo(RepoNode::new("api", dir.join("api")).with_branch("main"))
            .unwrap();
        host.workspace_add_repo(RepoNode::new("docs", dir.join("docs")))
            .unwrap();
        host.workspace_add_edge("web", "api", WorkspaceEdgeKind::ConsumesApiFrom)
            .unwrap();
        host.workspace_add_edge("api", "docs", WorkspaceEdgeKind::Documents)
            .unwrap();
        host.workspace_add_edge("web", "api", WorkspaceEdgeKind::DependsOn)
            .unwrap();
        let graph = host.workspace_graph();
        let repo_ids: Vec<&str> = graph.repos.iter().map(|r| r.repo_id.as_str()).collect();
        assert_eq!(repo_ids, vec!["api", "docs", "web"]);
        assert_eq!(graph.edges.len(), 3);
        let edge_tuples: Vec<(&str, &str, WorkspaceEdgeKind)> = graph
            .edges
            .iter()
            .map(|e| (e.from.as_str(), e.to.as_str(), e.kind))
            .collect();
        assert_eq!(
            edge_tuples,
            vec![
                ("api", "docs", WorkspaceEdgeKind::Documents),
                ("web", "api", WorkspaceEdgeKind::ConsumesApiFrom),
                ("web", "api", WorkspaceEdgeKind::DependsOn),
            ]
        );
        assert_eq!(host.workspace_graph(), graph);
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        assert_eq!(reopened.workspace_graph(), graph);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn environment_switch_records_durable_event_and_session_continues() {
        use crate::services::EnvironmentNode;
        let dir = std::env::temp_dir().join(format!("hide_ws_env_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        host.workspace_add_environment(
            EnvironmentNode::new("dev")
                .with_fs_roots(vec![dir.join("web")])
                .with_tool_scopes(vec!["fs.read".to_string()]),
        )
        .unwrap();
        host.workspace_add_environment(
            EnvironmentNode::new("ci")
                .with_fs_roots(vec![dir.join("api")])
                .with_tool_scopes(vec!["fs.read".to_string(), "shell.run".to_string()]),
        )
        .unwrap();
        let first = host
            .environment_switch(session.clone(), "dev", "start local work")
            .await
            .unwrap();
        assert_eq!(first.previous_env, None);
        assert_eq!(first.new_env, "dev");
        assert_eq!(first.reason, "start local work");
        assert_eq!(first.tool_scopes, vec!["fs.read".to_string()]);
        let second = host
            .environment_switch(session.clone(), "ci", "run the suite")
            .await
            .unwrap();
        assert_eq!(second.previous_env.as_deref(), Some("dev"));
        assert_eq!(second.new_env, "ci");
        assert_eq!(second.reason, "run the suite");
        let switches = host.environment_switches(&session).await.unwrap();
        assert_eq!(switches.len(), 2);
        assert_eq!(switches[0].previous_env, None);
        assert_eq!(switches[0].new_env, "dev");
        assert_eq!(switches[1].previous_env.as_deref(), Some("dev"));
        assert_eq!(switches[1].new_env, "ci");
        assert_eq!(host.services.session(), session);
        host.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "still here" }),
            ))
            .await
            .unwrap();
        let events = host
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert_eq!(events.len(), 3);
 assert_eq!( events .iter() .filter(|e| e.kind == "environment.switch") .count(), 2 );
 assert!(host .environment_switch(session.clone(), "ghost", "nope") .await .is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn untrusted_repo_is_inert_until_trust_is_set() {
        use crate::services::{RepoNode, TrustState};
        let dir = std::env::temp_dir().join(format!("hide_ws_trust_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        host.workspace_add_repo(
            RepoNode::new("vendor", dir.join("vendor"))
                .with_instructions_ref("blob:instructions")
                .with_policy_ref("blob:policy"),
        )
        .unwrap();
        let untrusted = host.workspace_repo("vendor").unwrap();
        assert_eq!(untrusted.trust, TrustState::Untrusted);
        assert!(untrusted.instructions_ref.is_some());
        assert!(untrusted.policy_ref.is_some());
        assert_eq!(untrusted.active_instructions_ref(), None);
        assert_eq!(untrusted.active_policy_ref(), None);
        let trusted = host
            .workspace_set_repo_trust("vendor", TrustState::Trusted)
            .unwrap()
            .expect("the repo exists");
        assert_eq!(trusted.trust, TrustState::Trusted);
 assert_eq!( trusted.active_instructions_ref(), Some("blob:instructions") );
        assert_eq!(trusted.active_policy_ref(), Some("blob:policy"));
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        let after = reopened.workspace_repo("vendor").unwrap();
        assert_eq!(after.trust, TrustState::Trusted);
        assert_eq!(after.active_instructions_ref(), Some("blob:instructions"));
 assert!(reopened .workspace_set_repo_trust("ghost", TrustState::Trusted) .unwrap() .is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn the_trust_intent_enters_the_folder_into_the_graph() {
        use crate::services::TrustState;
        let dir = std::env::temp_dir().join(format!("hide_ws_trust_add_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let root = dir.join("vendor");
        host.handle_memory_workspace_env_intent(
            "workspace_set_repo_trust",
            &json!({
                "repo_id": "vendor",
                "root_path": root.to_string_lossy(),
                "trust": "trusted",
            }),
        )
        .await
        .expect("the folder enters the graph and the decision lands on it");
        let repo = host.workspace_repo("vendor").expect("the node was created");
        assert_eq!(repo.trust, TrustState::Trusted);
        assert_eq!(repo.root_path, root);
        let err = host
            .handle_memory_workspace_env_intent(
                "workspace_set_repo_trust",
                &json!({ "repo_id": "ghost", "trust": "trusted" }),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("root_path"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }
    fn dirty_source() -> String {
        let mut src = String::new();
        src.push_str("pub fn parse_port(raw: &str) -> u16 {\n");
        src.push_str("    raw.parse::<u16>().unwrap()\n");
        src.push_str("}\n\n");
        src.push_str("pub fn not_done() {\n");
        src.push_str("    todo!()\n");
        src.push_str("}\n\n");
        src.push_str("pub fn sprawling() {\n");
        for i in 0..90 {
            src.push_str(&format!("    let _v{i} = {i};\n"));
        }
        src.push_str("}\n");
        src
    }
    #[tokio::test]
    async fn run_static_analysis_fails_on_planted_issues_and_records_durable_receipt() {
        use hide_kernel::verify_plane::CheckKind;
        let dir = std::env::temp_dir().join(format!("hide_verify_dirty_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let sources = vec![SourceFile::new("src/net.rs", dirty_source())];
        let receipt = host
            .run_static_analysis(session.clone(), sources)
            .await
            .unwrap();
 assert!( receipt.verdict().is_fail(), "planted issues must fail the deterministic gate" );
        assert!(!receipt.is_pass());
        assert_eq!(receipt.receipt.tier, VerificationTier::Tier1Deterministic);
        assert_eq!(receipt.receipt.oracle, "static_analysis");
 assert_eq!( receipt.receipt.command, None, "an in-process oracle runs no command" );
        assert_eq!(receipt.receipt.scope, vec!["src/net.rs".to_string()]);
        assert!(!receipt.receipt.source_hash.is_empty());
        let kinds: std::collections::HashSet<CheckKind> =
            receipt.findings.iter().map(|f| f.check).collect();
        assert!(kinds.contains(&CheckKind::UnwrapOutsideTest));
 assert!( kinds.contains(&CheckKind::PanicMarker), "marker-macro finding expected" );
 assert!( kinds.contains(&CheckKind::LongFunction), "long-function finding expected" );
        let stored = host.verification_receipts(&session).await.unwrap();
        assert_eq!(stored.len(), 1, "exactly one receipt was recorded");
        assert_eq!(stored[0], receipt, "the stored receipt round-trips exactly");
        assert!(stored[0].verdict().is_fail());
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn run_static_analysis_passes_on_clean_source() {
        let dir = std::env::temp_dir().join(format!("hide_verify_clean_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let clean = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let receipt = host
            .run_static_analysis(
                session.clone(),
                vec![SourceFile::new("src/math.rs", clean)],
            )
            .await
            .unwrap();
        assert!(receipt.is_pass(), "clean source passes the deterministic gate");
        assert!(receipt.findings.is_empty());
        assert_eq!(receipt.findings_summary(), "no findings");
        let stored = host.verification_receipts(&session).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert!(stored[0].is_pass());
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn review_role_profiles_are_data_and_call_no_model() {
        let dir = std::env::temp_dir().join(format!("hide_verify_roles_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let profiles = host.review_role_profiles();
        assert_eq!(profiles.len(), 8, "all eight review roles are present");
        let correctness = host.review_role_profile(ReviewRole::Correctness);
        assert_eq!(correctness.role, ReviewRole::Correctness);
        assert!(!correctness.focus.is_empty());
        assert!(!correctness.acceptance.is_empty());
        assert!(correctness.output_schema_ref.starts_with("hide.review."));
        assert!(profiles.iter().any(|p| p.role == ReviewRole::Security));
        let session = host.services.session();
        let events = host
            .services
            .event_log
            .scan(Some(session), None, None)
            .await
            .unwrap();
 assert!( events.is_empty(), "review-role profiles perform no model call and emit no events" );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn probabilistic_review_cannot_override_a_failing_deterministic_receipt() {
        let dir = std::env::temp_dir().join(format!("hide_verify_authority_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let receipt = host
            .run_static_analysis(
                session.clone(),
                vec![SourceFile::new("src/net.rs", dirty_source())],
            )
            .await
            .unwrap();
        assert!(receipt.verdict().is_fail());
        let scope = receipt.receipt.scope.clone();
        let review = TieredVerdict::new(
            VerificationTier::Tier4Review,
            "correctness",
            hide_kernel::verify_plane::Verdict::Pass,
        );
        let decision =
            host.reconcile_review_for_scope(&scope, &[receipt.clone()], &[review.clone()]);
        assert!(matches!(decision, GateDecision::Reject { .. }));
        assert!(!hide_kernel::verify_plane::probabilistic_can_override_deterministic());
        let other = host.reconcile_review_for_scope(
            &["src/unrelated.rs".to_string()],
            &[receipt],
            &[review],
        );
        assert!(matches!(other, GateDecision::Inconclusive));
        let _ = std::fs::remove_dir_all(dir);
    }
    fn memory_test_host(label: &str) -> (PathBuf, BackendHost) {
        let dir = std::env::temp_dir().join(format!("hide_mem_{label}_{}", now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.user_root = dir.join("user_home");
        config.security.shell_default = Decision::Allow;
        config.security.workspace_write_default = Decision::Allow;
        let host =
            BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        (dir, host)
    }
    #[tokio::test]
    async fn production_submit_turn_writes_episodic_with_provenance() {
        use hawking_context::MemoryClass;
        use hide_core::api::Intent;
        let (dir, host) = memory_test_host("epi");
        let session = host.services.session();
        let marker = format!("episodic-marker-{}", now_ms());
        let ack = host
            .handle_intent(Intent::SubmitTurn {
                session_id: session.clone(),
                text: marker.clone(),
                attachments: vec![],
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let episodes = host
            .services
            .classed_memory
            .list_class(MemoryClass::Episodic)
            .unwrap();
        let hit = episodes
            .iter()
            .find(|r| r.text.contains(&marker))
            .expect("submit_turn must write an episodic record with the prompt text");
        assert_eq!(hit.provenance.writer, "event_stream");
        assert_eq!(hit.session_id.as_deref(), Some(session.as_str()));
        assert!(hit.provenance.written_at_ms > 0);
 assert!(hit .provenance .evidence .iter() .any(|e| e.starts_with("event_id:")));
 assert_eq!( host.services .classed_memory .count(MemoryClass::Verification) .unwrap(), 0 );
 assert_eq!( host.services .classed_memory .count(MemoryClass::User) .unwrap(), 0 );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn production_tool_receipt_procedural_success_only() {
        use hawking_context::MemoryClass;
        use hide_core::tool::{DispatchObserver, ToolError, ToolResult};
        use hide_core::types::EffectSet;
        let (dir, host) = memory_test_host("proc");
        let session = host.services.session();
        let recorder = DispatchRecorder::new(host.services.clone(), host.ui_bus().clone());
        let ok_call = ToolCall::new(
            "shell.run",
            json!({ "argv": ["cargo", "test", "-p", "hide-core"] }),
        );
        let mut ok = ToolResult::ok(
            ok_call.call_id.clone(),
            Some(json!({ "stdout": "test result: ok. 1 passed" })),
            EffectSet::default(),
        );
        ok.exit_code = Some(0);
        recorder.after(&ok_call, None, &ok).await;
        let rows = host
            .services
            .classed_memory
            .list_class(MemoryClass::Procedural)
            .unwrap();
        assert_eq!(rows.len(), 1, "successful receipt must write one recipe");
        assert!(rows[0].text.contains("cargo test"));
        assert_eq!(rows[0].provenance.writer, "tool_receipt");
        assert_eq!(rows[0].session_id.as_deref(), Some(session.as_str()));
 assert!( host.services .classed_memory .count(MemoryClass::SemanticProject) .unwrap() >= 1 );
 assert_eq!( host.services .classed_memory .count(MemoryClass::User) .unwrap(), 0 );
 assert_eq!( host.services .classed_memory .count(MemoryClass::Verification) .unwrap(), 0 );
        let before = rows.len();
        let proj_before = host
            .services
            .classed_memory
            .count(MemoryClass::SemanticProject)
            .unwrap();
        let fail_call = ToolCall::new("shell.run", json!({ "argv": ["false"] }));
        let mut fail = ToolResult::ok(fail_call.call_id.clone(), None, EffectSet::default());
        fail.status = ToolStatus::ToolError;
        fail.ok = false;
        fail.error = Some(ToolError::new("EXEC_FAILED", "boom", false));
        recorder.after(&fail_call, None, &fail).await;
        assert_eq!(host.services .classed_memory .count(MemoryClass::Procedural) .unwrap(), before);
        assert_eq!(host.services .classed_memory .count(MemoryClass::SemanticProject) .unwrap(), proj_before);
        let mut sandbox_fail = ToolResult::ok(
            ToolCall::new("shell.run", json!({ "argv": ["true"] })).call_id,
            Some(json!({ "exit_code": 71, "stderr": "sandbox-exec: Operation not permitted" })),
            EffectSet::default(),
        );
        sandbox_fail.exit_code = Some(71);
        let nz_call = ToolCall::new("shell.run", json!({ "argv": ["true"] }));
        recorder.after(&nz_call, None, &sandbox_fail).await;
        assert_eq!(host.services .classed_memory .count(MemoryClass::Procedural) .unwrap(), before);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn production_verifier_writes_verification_model_turn_does_not_write_protected() {
        use hawking_context::MemoryClass;
        use hawking_orch::inference::{InferenceClient, StubInferenceClient};
        let (dir, host) = memory_test_host("ver");
        let session = host.services.session();
        let services = host.services.clone();
        let clean = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let receipt = host
            .run_static_analysis(
                session.clone(),
                vec![SourceFile::new("src/math.rs", clean)],
            )
            .await
            .unwrap();
        assert!(receipt.is_pass());
        let vrows = services
            .classed_memory
            .list_class(MemoryClass::Verification)
            .unwrap();
        assert_eq!(vrows.len(), 1);
        assert_eq!(vrows[0].provenance.writer, "verifier");
        assert_eq!(vrows[0].evidence_tier.as_deref(), Some("proven"));
        let before_user = services.classed_memory.count(MemoryClass::User).unwrap();
        let before_ver = services
            .classed_memory
            .count(MemoryClass::Verification)
            .unwrap();
        let inference: Arc<dyn InferenceClient> =
            Arc::new(StubInferenceClient::new("model says hello"));
        let ui_bus = Arc::new(UiEventBus::default());
        let _ = run_turn_core(
            inference,
            services.event_log.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            ui_bus,
            session.clone(),
            "please do not forge verification".into(),
            None,
            Some("run-model-turn".into()),
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();
        assert_eq!(services.classed_memory.count(MemoryClass::User).unwrap(), before_user);
        assert_eq!(services .classed_memory .count(MemoryClass::Verification) .unwrap(), before_ver);
        let draft = MemoryDraft::new(
            MemoryScope::User("person-1".into()),
            "prefer snake_case in rust",
            "settings",
            "user",
        );
        host.memory_add(draft).unwrap();
 assert_eq!( services.classed_memory.count(MemoryClass::User).unwrap(), before_user + 1 );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn production_write_then_compile_round_trip() {
        use hawking_context::compiler::{CompileInput, ContextCompiler};
        use hawking_context::profiles::ContextProfile;
        use hawking_context::sources::ClassedMemoryContextSource;
        use hawking_context::{ClassBudgets, MemoryClass};
        use hide_core::api::Intent;
        use hide_core::ids::ModelId;
        use hide_core::runtime::{ModelArchitecture, ModelDescriptor};
        let (dir, host) = memory_test_host("rt");
        let session = host.services.session();
        let marker = format!("roundtrip-live-{}", now_ms());
        let ack = host
            .handle_intent(Intent::SubmitTurn {
                session_id: session.clone(),
                text: marker.clone(),
                attachments: vec![],
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        {
            use hide_core::tool::{DispatchObserver, ToolResult};
            use hide_core::types::EffectSet;
            let recorder = DispatchRecorder::new(host.services.clone(), host.ui_bus().clone());
            let call = ToolCall::new(
                "shell.run",
                json!({ "argv": ["cargo", "test"], "marker": &marker }),
            );
            let mut ok = ToolResult::ok(
                call.call_id.clone(),
                Some(json!({ "stdout": format!("ok {marker}") })),
                EffectSet::default(),
            );
            ok.exit_code = Some(0);
            recorder.after(&call, None, &ok).await;
        }
        let budgets = ClassBudgets::default_small();
        let mut compiler = ContextCompiler::new();
        compiler.add_source(
            ClassedMemoryContextSource::new(host.services.classed_memory.clone(), budgets)
                .with_session(session.as_str()),
        );
        let model = ModelDescriptor {
            id: ModelId::new(),
            name: "test".into(),
            architecture: ModelArchitecture::Transformer,
            context_tokens: 2048,
            tokenizer_signature: "test".into(),
            footprint_mb: 1,
        };
        let compiled = compiler
            .compile(CompileInput {
                profile: ContextProfile::coding_default(2048),
                model,
                task: marker.clone(),
            })
            .await
            .unwrap();
        assert!(compiled.prompt.contains(&marker));
        let ret = host
            .services
            .classed_memory
            .last_retrieval()
            .expect("compile ran retrieve");
        assert!(!ret.slice(MemoryClass::Episodic).unwrap().hits.is_empty());
        assert!(!ret.slice(MemoryClass::Procedural).unwrap().hits.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }
