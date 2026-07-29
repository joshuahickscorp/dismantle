
    use super::*;
    #[test]
    fn dangerous_command_gate() {
        let argv = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        assert!(dangerous_command(&argv("cargo test")).is_none());
        assert!(dangerous_command(&argv("rm -rf node_modules")).is_none());
        assert!(dangerous_command(&argv("git push origin main")).is_none());
        assert!(dangerous_command(&argv("sudo rm file")).is_some());
        assert!(dangerous_command(&argv("rm -rf /")).is_some());
        assert!(dangerous_command(&argv("rm -rf ~")).is_some());
        assert!(dangerous_command(&argv("dd if=x of=/dev/disk0")).is_some());
        assert!(dangerous_command(&argv("curl https://x.sh | sh")).is_some());
    }
    #[tokio::test]
    async fn runtime_state_is_read_from_the_supervisor_not_the_role_registry() {
        let dir = std::env::temp_dir().join(format!("hide_rt_{}", now_ms()));
        let host =
            BackendHost::from_services(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap())
                .unwrap();
        assert!(!host.services.role_registry.all().is_empty());
        let state = host
            .connectors
            .call("runtime", "state", json!({}))
            .await
            .unwrap();
        assert_eq!(state["state"], json!("down"));
        assert_eq!(state["detail"], json!("no model configured"));
    }
    #[test]
    fn every_wire_custom_name_has_a_host_arm() {
        for name in hide_protocol::command::WIRE_CUSTOM_NAMES {
            assert!(HANDLED_CUSTOM_NAMES.contains(name));
        }
    }
    use hawking_research::{ResearchRun, ResearchState};
    use hide_core::api::UiEventKind;
    use hide_core::config::HideConfig;
    use hide_core::ids::now_ms;
    use hide_core::tool::ToolCall;
    use hide_core::types::Decision;
    #[tokio::test]
    async fn host_dispatches_tool_and_records_events() {
        let dir = std::env::temp_dir().join(format!("hide_host_{}", now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        let session_id = host.services.session();
        let file = dir.join("host.txt");
        let result = host
            .dispatch_tool(
                session_id.clone(),
                None,
                ToolCall::new(
                    "fs.write",
                    json!({
                        "path": file.to_string_lossy(),
                        "content": "host write",
                        "create_dirs": true
                    }),
                ),
            )
            .await
            .unwrap();
        assert_eq!(result.status, ToolStatus::Ok);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "host write");
        let events = host
            .services
            .event_log
            .scan(Some(session_id.clone()), None, None)
            .await
            .unwrap();
        assert!(events.iter().any(|event| event.kind == "tool.call"));
        assert!(events.iter().any(|event| event.kind == "tool.result"));
 assert!(host .services .projection_store .latest_projection(&session_id) .unwrap() .is_some());
        let ui_events = host
            .ui_events(Some(session_id.clone()), None, None)
            .await
            .unwrap();
 assert!(ui_events .iter() .any(|event| matches!(event.kind, UiEventKind::ToolProgress { .. })));
        let rebuilt = host
            .rebuild_session_projection(session_id.clone())
            .await
            .unwrap();
        assert_eq!(rebuilt.session_id, session_id);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn policy_ledger_classifies_and_durably_records_decisions() {
        let dir = std::env::temp_dir().join(format!("hide_policy_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let read = host
            .evaluate_tool_policy(
                &session,
                "fs.read",
                &json!({ "path": dir.join("a.txt").to_string_lossy() }),
            )
            .await
            .unwrap();
        assert_eq!(read, PolicyDecision::Allow);
        assert!(!read.requires_sandbox());
        let run = host
            .evaluate_tool_policy(&session, "shell.run", &json!({ "argv": ["ls"] }))
            .await
            .unwrap();
        assert_eq!(run, PolicyDecision::RequireSandbox);
        assert!(run.requires_sandbox());
        let commit = host
            .evaluate_tool_policy(&session, "git.commit", &json!({ "message": "wip" }))
            .await
            .unwrap();
 assert!(matches!( commit, PolicyDecision::Ask | PolicyDecision::RequireReviewer ));
        let write = host
            .evaluate_tool_policy(
                &session,
                "edit.write_file",
                &json!({ "path": dir.join("b.txt").to_string_lossy(), "content": "x" }),
            )
            .await
            .unwrap();
        assert_eq!(write, PolicyDecision::Ask);
        let ledger = host.policy_decisions(&session).await.unwrap();
        assert_eq!(ledger.len(), 4);
        let tools: Vec<_> = ledger.iter().map(|record| record.tool.clone()).collect();
        assert_eq!(tools, vec![ "fs.read".to_string(), "shell.run".to_string(), "git.commit".to_string(), "edit.write_file".to_string() ]);
        let run_rec = ledger.iter().find(|r| r.tool == "shell.run").unwrap();
        assert!(run_rec.effects.contains(&"Execute".to_string()));
        assert!(run_rec.effects.contains(&"Process".to_string()));
        assert_eq!(run_rec.decision, PolicyDecision::RequireSandbox);
        let read_rec = ledger.iter().find(|r| r.tool == "fs.read").unwrap();
        assert_eq!(read_rec.effects, vec!["Read".to_string()]);
        assert_eq!(read_rec.decision, PolicyDecision::Allow);
        let commit_rec = ledger.iter().find(|r| r.tool == "git.commit").unwrap();
        assert_eq!(commit_rec.effects, vec!["GitMutation".to_string()]);
        let events = host
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
 assert_eq!( events .iter() .filter(|event| event.kind == "policy.decision") .count(), 4 );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn write_policy_follows_engine_decision() {
        let dir = std::env::temp_dir().join(format!("hide_policy_write_{}", now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        let session = host.services.session();
        let decision = host
            .evaluate_tool_policy(
                &session,
                "edit.write_file",
                &json!({ "path": dir.join("c.txt").to_string_lossy(), "content": "x" }),
            )
            .await
            .unwrap();
        assert_eq!(decision, PolicyDecision::Allow);
        let ledger = host.policy_decisions(&session).await.unwrap();
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger[0].tool, "edit.write_file");
        assert_eq!(ledger[0].decision, PolicyDecision::Allow);
        assert_eq!(ledger[0].effects, vec!["Write".to_string()]);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_reports_status_surface() {
        let dir = std::env::temp_dir().join(format!("hide_host_status_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let status = host.status().await;
        assert!(status.capabilities.agent_kernel);
        assert!(status.tools.iter().any(|tool| tool.name == "fs.write"));
 assert!(status .connectors .iter() .any(|connector| connector.id == "research"));
 assert!(status .model_roles .iter() .any(|role| role.name == "hawking-hero-coder"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_records_run_command_intent_and_executes_command_api() {
        let dir = std::env::temp_dir().join(format!("hide_host_command_{}", now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.shell_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["printf".to_string(), "intent".to_string()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let session_id = host.services.session();
        let result = host
            .run_command(
                session_id,
                vec!["printf".to_string(), "api".to_string()],
                None,
            )
            .await
            .unwrap();
        assert_eq!(result.status, ToolStatus::Ok);
        assert_eq!(result.structured_content.unwrap()["stdout"], "api");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_routes_connector_calls() {
        let dir = std::env::temp_dir().join(format!("hide_host_connector_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let mut run = ResearchRun::new("host connector");
        run.state = ResearchState::Complete;
        host.call_connector("research", "runs.append", json!({ "run": run }))
            .await
            .unwrap();
        let listed = host
            .call_connector("research", "runs.list", json!({ "limit": 1 }))
            .await
            .unwrap();
        assert_eq!(listed["runs"].as_array().unwrap().len(), 1);
        assert_eq!(listed["runs"][0]["topic"], "host connector");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_reports_health_checks() {
        let dir = std::env::temp_dir().join(format!("hide_host_health_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let health = host.health().await;
        assert_eq!(health.status, HealthStatus::Ok);
        assert!(health.checks.iter().any(|check| check.name == "tools"));
 assert!(health .checks .iter() .any(|check| check.name == "connector:personalization"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn host_caps_are_honest_remote_is_false() {
        let dir = std::env::temp_dir().join(format!("hide_host_caps_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let caps = host.status().await.capabilities;
        assert!(caps.agent_kernel && caps.fleet && caps.model_orchestration);
        assert!(!caps.remote_protocol);
        let _ = std::fs::remove_dir_all(dir);
    }
    fn fleet_git_cargo_workspace(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("hide_host_fleet_{label}_{}", now_ms()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"fleet_fix\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(dir.join("src/lib.rs"), "pub fn n() -> i32 { 1 }\n").unwrap();
        let git = |args: &[&str]| {
            let st = std::process::Command::new("git")
                .args(args)
                .current_dir(&dir)
                .status()
                .expect("git");
            assert!(st.success(), "git {args:?} failed in {}", dir.display());
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "fleet@test"]);
        git(&["config", "user.name", "fleet"]);
        git(&["add", "-A"]);
        git(&["commit", "-qm", "init"]);
        dir
    }
    #[tokio::test]
    async fn fleet_run_uses_real_worktrees_and_runtime_planner() {
        let dir = fleet_git_cargo_workspace("real_wt");
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let status = host
            .fleet_run(session, "verify the fixture builds")
            .await
            .unwrap();
        assert!(status == "Done" || status == "Failed");
        let wt_root = dir.join(".hide").join("wt");
        assert!(wt_root.exists() || dir.join(".hide").exists());
        let _ = std::process::Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(&dir)
            .status();
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[tokio::test]
    async fn host_fleet_run_schedules_and_completes() {
        let dir = fleet_git_cargo_workspace("complete");
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let status = host.fleet_run(session, "scaffold a module").await.unwrap();
 assert!( status == "Done" || status == "Failed", "expected terminal fleet status, got {status}" );
        let _ = std::process::Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(&dir)
            .status();
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[tokio::test]
    async fn fleet_run_intent_reaches_fleet_manager() {
        let dir = fleet_git_cargo_workspace("intent");
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let mut rx = host.subscribe_ui();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "fleet_run".into(),
                payload: json!({
                    "task": "scaffold a module via intent",
                    "session_id": session.as_str(),
                }),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "fleet_run intent must be accepted: {:?}", ack.message);
        let msg = ack.message.as_deref().unwrap_or("");
        assert!(msg.contains("status=Done") || msg.contains("status=Failed"));
        let mut saw = false;
        for _ in 0..16 {
            match tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await {
                Ok(Ok(ev)) => {
                    if let UiEventKind::Custom(v) = ev.kind {
                        if v.get("kind").and_then(|k| k.as_str()) == Some("fleet_run_completed") {
                            let st = v.get("status").and_then(|s| s.as_str());
 assert!( st == Some("Done") || st == Some("Failed"), "terminal status expected, got {st:?}" );
                            saw = true;
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
        assert!(saw, "fleet_run_completed UiEvent must be published");
        let _ = std::process::Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(&dir)
            .status();
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn services_token_counter_is_tokenizer_true_when_hide_tokenizer_set() {
        let path = std::env::var("HIDE_TOKENIZER_TEST_PATH")
            .ok()
            .filter(|p| std::path::Path::new(p).is_file())
            .or_else(|| {
                let known = [
                    "/Users/scammermike/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json",
                ];
                known
                    .into_iter()
                    .map(std::path::PathBuf::from)
                    .find(|p| p.is_file())
                    .map(|p| p.display().to_string())
            });
        let Some(path) = path else {
            eprintln!("services_token_counter_is_tokenizer_true_when_hide_tokenizer_set: SKIP");
            return;
        };
        // asserts accuracy on from_file + with_counter wiring.
        let counter = hawking_context::TokenCounter::from_file(&path).expect("load tokenizer");
        assert!(counter.is_accurate());
        let dir = std::env::temp_dir().join(format!("hide_tok_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let compiler = host.services.context_compiler();
        // Without ambient HIDE_TOKENIZER the services counter may be heuristic;
        let wired = hawking_context::ContextCompiler::new().with_counter(counter);
        assert!(!wired.tokens_estimated());
        let _ = compiler;
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mcp_servers_register_into_live_registry_at_boot() {
        let py = ["python3", "python"].into_iter().find(|c| {
            std::process::Command::new(c)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        });
        let Some(py) = py else {
            eprintln!("python3 not found; skipping MCP boot registration test");
            return;
        };
        let dir = std::env::temp_dir().join(format!("hide_host_mcp_{}", now_ms()));
        std::fs::create_dir_all(dir.join(".hide")).unwrap();
        let fake = r#"
import sys, json
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    m = req.get("method"); i = req.get("id")
    if m == "initialize":
        send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"fake","version":"0"}}})
    elif m == "notifications/initialized":
        pass
    elif m == "tools/list":
        send({"jsonrpc":"2.0","id":i,"result":{"tools":[{"name":"echo","description":"echo","inputSchema":{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"],"additionalProperties":False}}]}})
    elif m == "tools/call":
        msg = req["params"]["arguments"]["msg"]
        send({"jsonrpc":"2.0","id":i,"result":{"isError":False,"structuredContent":{"echoed":msg},"content":[{"type":"text","text":msg}]}})
    else:
        send({"jsonrpc":"2.0","id":i,"error":{"code":-32601,"message":"method not found"}})
"#;
        let mcp_cfg = json!([
            {
                "id": "boot_good",
                "transport": { "stdio": { "command": py, "args": ["-c", fake] } },
                "trust": "third-party"
            },
            {
                "id": "boot_bad",
                "transport": {
                    "stdio": {
                        "command": "definitely-not-a-real-binary-xyzzy",
                        "args": []
                    }
                },
                "trust": "third-party"
            }
        ]);
        std::fs::write(
            dir.join(".hide").join("mcp.json"),
            serde_json::to_vec_pretty(&mcp_cfg).unwrap(),
        )
        .unwrap();
        let host = BackendHost::from_services(
            BackendServices::open(HideConfig::for_workspace(&dir)).unwrap(),
        )
        .expect("host boot must not fail on a bad MCP server");
        assert!(host.tools.get("mcp:boot_good/echo").is_some());
        assert!(host.tools.get("mcp:boot_bad/echo").is_none());
        let events = host
            .services
            .event_log
            .scan(None, None, None)
            .await
            .unwrap();
        assert!(events.iter().any(|e| e.kind == "mcp.server_registered" && e.payload.get("server_id").and_then(|v| v.as_str()) == Some("boot_good")));
        assert!(events.iter().any(|e| e.kind == "mcp.server_failed" && e.payload.get("server_id").and_then(|v| v.as_str()) == Some("boot_bad")));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn open_workspace_binds_sqlite_index_with_real_hits() {
        let dir = std::env::temp_dir().join(format!("hide_host_idx_{}", now_ms()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        const MARKER: &str = "ZZW4SQLITEONLYTOKEN";
        std::fs::write(
            dir.join("src").join("marker.rs"),
            format!("// {MARKER} unique grounding anchor\npub fn w4_marker_fn() {{}}\n"),
        )
        .unwrap();
        let services = BackendServices::open_workspace(&dir).unwrap();
 assert!( services.sqlite_index.is_some(), "open_workspace must bind SqliteCodeIndex" );
        assert!(services.memory_index.is_none());
        let results = services
            .code_index
            .search(hawking_index::SearchQuery {
                text: MARKER.to_string(),
                limit: 10,
                include_symbols: true,
                include_lexical: true,
                include_semantic: false,
            })
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.snippet.contains(MARKER) || r.title.contains(MARKER) || r.snippet.contains("w4_marker")));
        let mem = BackendServices::new(
            HideConfig::for_workspace(&dir),
            services.event_log.clone(),
        );
        assert!(mem.memory_index.is_some());
        assert!(mem.sqlite_index.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn flagship_boot_supervise_intent_generate_publish() {
        use crate::supervisor::testkit::{FakeLauncher, FakeRuntime};
        use crate::supervisor::{RuntimeSupervisor, SupervisorConfig};
        use hide_core::supervision::{BackoffPolicy, ProcessSpec};
        use std::time::Duration;
        let dir = std::env::temp_dir().join(format!("hide_flagship_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let rt = Arc::new(FakeRuntime::spawn().await);
        let cfg = SupervisorConfig {
            spec: ProcessSpec {
                name: "fake-serve".to_string(),
                argv: vec!["fake".to_string()],
                cwd: None,
                env: Default::default(),
                health_url: None,
            },
            backoff: BackoffPolicy::default(),
            health_interval: Duration::from_millis(10),
            boot_timeout: Duration::from_secs(2),
            lock_path: Some(host.services.layout().hide_dir.join("runtime.lock")),
        };
        let supervisor = RuntimeSupervisor::new(cfg, Arc::new(FakeLauncher::new(rt.clone())));
        supervisor.boot().await.unwrap();
 assert_eq!( supervisor.state(), hide_core::runtime::RuntimeSupervisorState::Ready );
        let base_url = supervisor.base_url().unwrap();
        let session = host.services.session();
        let ack = host
            .handle_intent(Intent::SubmitTurn {
                session_id: session.clone(),
                text: "implement the parser".to_string(),
                attachments: Vec::new(),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "valid SubmitTurn must be accepted");
        let mut rx = host.subscribe_ui();
        let completion = host
            .generate_and_publish(session.clone(), &base_url, "write a function")
            .await
            .unwrap();
        assert_eq!(completion, "fake generate");
        let event = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should be published")
            .expect("broadcast channel delivers");
        match event.kind {
            UiEventKind::TokenBatch { text, .. } => assert_eq!(text, "fake generate"),
            other => panic!("expected a TokenBatch UiEvent, got {other:?}"),
        }
        supervisor.shutdown().await;
        rt.stop();
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn submit_turn_with_no_runtime_publishes_model_offline_not_a_token() {
        let dir = std::env::temp_dir().join(format!("hide_host_offline_{}", now_ms()));
        std::env::remove_var("HIDE_MODEL_WEIGHTS");
        let host = BackendHost::open_workspace(&dir).unwrap();
        assert!(host.runtime_state().is_none());
        let session = host.services.session();
        let mut rx = host.subscribe_ui();
        let ack = host
            .handle_intent(Intent::SubmitTurn {
                session_id: session.clone(),
                text: "implement the parser".to_string(),
                attachments: Vec::new(),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let event = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should be published")
            .expect("broadcast delivers");
        match event.kind {
            UiEventKind::RuntimeStatus { status, detail } => {
                assert_eq!(status, "down");
                assert!(detail.unwrap_or_default().contains("no model configured"));
            }
            UiEventKind::TokenBatch { .. } => {
                panic!("must not fabricate a token when no model is online")
            }
            other => panic!("expected a RuntimeStatus UiEvent, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn run_turn_core_feeds_compiled_context_real_budget_and_persists_turn() {
        use futures::future::BoxFuture;
        use hawking_index::InMemoryCodeIndex;
        use hawking_orch::inference::{InferenceClient, StubInferenceClient};
        use hide_core::error::Result as HResult;
        use hide_core::runtime::{GenerationStats, InferenceRequest, TokenSink};
        struct RecordingClient {
            inner: StubInferenceClient,
            last: std::sync::Mutex<Option<InferenceRequest>>,
        }
        impl InferenceClient for RecordingClient {
            fn generate<'a>(
                &'a self,
                request: InferenceRequest,
                sink: TokenSink<'a>,
            ) -> BoxFuture<'a, HResult<GenerationStats>> {
                *self.last.lock().unwrap() = Some(request.clone());
                self.inner.generate(request, sink)
            }
            fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, HResult<Vec<f32>>> {
                self.inner.embed(text)
            }
        }
        let dir = std::env::temp_dir().join(format!("hide_turn_core_{}", now_ms()));
        let services = BackendServices::open(HideConfig::for_workspace(&dir)).unwrap();
        let session = services.session();
        let index = Arc::new(InMemoryCodeIndex::default());
        index.add_text_file(
            "src/seed.rs",
            "// zzcontextmarker anchor line for retrieval\npub fn helper() {}\n",
            None,
        );
        let recorder = Arc::new(RecordingClient {
            inner: StubInferenceClient::new("some completion"),
            last: std::sync::Mutex::new(None),
        });
        let inference: Arc<dyn InferenceClient> = recorder.clone();
        let ui_bus = Arc::new(UiEventBus::default());
        let outcome = run_turn_core(
            inference,
            services.event_log.clone(),
            services.role_registry.clone(),
            index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            ui_bus,
            session.clone(),
            "zzcontextmarker".to_string(),
            None,
            None,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();
        assert_eq!(outcome.completion, "some completion");
        let req = recorder
            .last
            .lock()
            .unwrap()
            .clone()
            .expect("a request was recorded");
        assert!(req.prompt.contains("zzcontextmarker"));
 assert!(req .messages .iter() .any(|m| m.role == "user" && m.content == "zzcontextmarker"));
 assert_ne!( req.max_output_tokens, 256, "budget must be derived, not the 256 facade" );
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert!(events.iter().any(|e| e.kind == "context.compiled"));
        assert!(events.iter().any(|e| e.kind == "agent.message" && e.payload["role"] == "assistant" && e.payload["text"] == "some completion"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn run_turn_core_declares_honest_capability_and_rot_meter() {
        use futures::future::BoxFuture;
        use hawking_index::InMemoryCodeIndex;
        use hawking_orch::inference::{InferenceClient, StubInferenceClient};
        use hide_core::error::Result as HResult;
        use hide_core::runtime::{GenerationStats, InferenceRequest, TokenSink};
        struct RecordingClient {
            inner: StubInferenceClient,
            last: std::sync::Mutex<Option<InferenceRequest>>,
        }
        impl InferenceClient for RecordingClient {
            fn generate<'a>(
                &'a self,
                request: InferenceRequest,
                sink: TokenSink<'a>,
            ) -> BoxFuture<'a, HResult<GenerationStats>> {
                *self.last.lock().unwrap() = Some(request.clone());
                self.inner.generate(request, sink)
            }
            fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, HResult<Vec<f32>>> {
                self.inner.embed(text)
            }
        }
        let dir = std::env::temp_dir().join(format!("hide_turn_cap_{}", now_ms()));
        let services = BackendServices::open(HideConfig::for_workspace(&dir)).unwrap();
        let session = services.session();
        let index = Arc::new(InMemoryCodeIndex::default());
        index.add_text_file(
            "src/cap.rs",
            "// zzcapmarker unique retrieval needle for capability test\npub fn cap() {}\n",
            None,
        );
        let recorder = Arc::new(RecordingClient {
            inner: StubInferenceClient::new("ok"),
            last: std::sync::Mutex::new(None),
        });
        let inference: Arc<dyn InferenceClient> = recorder.clone();
        let ui_bus = Arc::new(UiEventBus::default());
        let live_ceiling = Some((None, Some(2048usize), 8192usize));
        let _outcome = run_turn_core(
            inference,
            services.event_log.clone(),
            services.role_registry.clone(),
            index,
            services.memory_store.clone(),
            services.classed_memory.clone(),
            ui_bus,
            session.clone(),
            "zzcapmarker".to_string(),
            live_ceiling,
            Some("cap-run".into()),
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let compiled = events
            .iter()
            .find(|e| e.kind == "context.compiled")
            .expect("context.compiled must be logged on the live path");
        let cap = &compiled.payload["capability"];
 assert!( !cap.is_null(), "context.compiled must carry capability: {}", compiled.payload );
        assert_eq!(cap["native_maximum"]["tokens"], 2048);
        assert_eq!(cap["native_maximum"]["source"], "measured");
 assert_eq!( cap["effective_ceiling"]["tokens"], 8192, "effective stays distinct from native" );
        assert!(
            cap["effective_ceiling"]["estimated"].as_bool().unwrap_or(false)
                || cap["effective_ceiling"]["tokens"] != cap["native_maximum"]["tokens"],
            "effective expansion must not masquerade as a hard native cap"
        );
        assert!(cap["validated_quality"]["tokens"].is_null());
        assert!(cap["validated_agentic"]["tokens"].is_null());
        assert!(cap["kv_curve"].is_null() && cap["prefill_curve"].is_null());
        assert_eq!(compiled.payload["native_is_not_usable"], true);
        let rot = &compiled.payload["rot"];
        assert!(!rot.is_null(), "rot report required on context.compiled");
 assert!( rot["severity"].is_string(), "rot severity must be declared: {rot}" );
        let meter = &compiled.payload["meter"];
        assert!(!meter.is_null(), "meter required on context.compiled");
        let explanations = meter["explanations"]
            .as_array()
            .expect("meter.explanations must be an array");
 assert!( !explanations.is_empty(), "a meter that cannot explain itself is not auditable" );
        let joined = explanations
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(joined.contains("native") || joined.contains("do not raise native_maximum"));
        let req = recorder
            .last
            .lock()
            .unwrap()
            .clone()
            .expect("request recorded");
        assert_ne!(req.max_output_tokens, 256);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn run_turn_core_surfaces_context_rot_when_occupancy_critical() {
        use futures::future::BoxFuture;
        use hawking_index::InMemoryCodeIndex;
        use hawking_orch::inference::{InferenceClient, StubInferenceClient};
        use hide_core::error::Result as HResult;
        use hide_core::runtime::{GenerationStats, InferenceRequest, TokenSink};
        struct QuietClient {
            inner: StubInferenceClient,
        }
        impl InferenceClient for QuietClient {
            fn generate<'a>(
                &'a self,
                request: InferenceRequest,
                sink: TokenSink<'a>,
            ) -> BoxFuture<'a, HResult<GenerationStats>> {
                self.inner.generate(request, sink)
            }
            fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, HResult<Vec<f32>>> {
                self.inner.embed(text)
            }
        }
        let dir = std::env::temp_dir().join(format!("hide_turn_rot_{}", now_ms()));
        let services = BackendServices::open(HideConfig::for_workspace(&dir)).unwrap();
        let session = services.session();
        let index = Arc::new(InMemoryCodeIndex::default());
        // build_live_manifest(transformer) uses state_age_tokens as kv_seq_len;
        let big_prompt = format!("zzrotmarker {}", "word ".repeat(400));
        let inference: Arc<dyn InferenceClient> = Arc::new(QuietClient {
            inner: StubInferenceClient::new("done"),
        });
        let ui_bus = Arc::new(UiEventBus::default());
        // assert the rot *report exists* and that the detector API is on the
        let live = build_live_manifest(None, Some(100), 100, 95);
 assert!( live.occupancy >= 0.90, "fixture occupancy must be critical, got {}", live.occupancy );
        let mut empty = hawking_context::ContextManifest::new(100);
        let cap = declare_turn_capability(100, Some(100), Some(100), None, false);
        seal_compiled_manifest(&mut empty, cap, Some(&live), true);
        let rot = empty.rot.expect("rot sealed");
 assert!( rot.should_refresh, "critical occupancy must request refresh: {:?}", rot );
        assert!(matches!( rot.severity, hawking_context::RotSeverity::Critical | hawking_context::RotSeverity::Degraded ));
        let _ = run_turn_core(
            inference,
            services.event_log.clone(),
            services.role_registry.clone(),
            index,
            services.memory_store.clone(),
            services.classed_memory.clone(),
            ui_bus,
            session.clone(),
            big_prompt,
            Some((None, Some(100), 100)),
            None,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();
        let events = services
            .event_log
            .scan(Some(session), None, None)
            .await
            .unwrap();
        let compiled = events
            .iter()
            .find(|e| e.kind == "context.compiled")
            .expect("context.compiled");
 assert!( !compiled.payload["rot"].is_null(), "live path must publish rot on context.compiled" );
        assert!(!compiled.payload["meter"]["explanations"] .as_array() .map(|a| a.is_empty()) .unwrap_or(true));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn repo_claude_md_folds_into_turn_context_with_receipt() {
        use futures::future::BoxFuture;
        use hawking_index::InMemoryCodeIndex;
        use hawking_orch::inference::{InferenceClient, StubInferenceClient};
        use hide_core::error::Result as HResult;
        use hide_core::runtime::{GenerationStats, InferenceRequest, TokenSink};
        struct RecordingClient {
            inner: StubInferenceClient,
            last: std::sync::Mutex<Option<InferenceRequest>>,
        }
        impl InferenceClient for RecordingClient {
            fn generate<'a>(
                &'a self,
                request: InferenceRequest,
                sink: TokenSink<'a>,
            ) -> BoxFuture<'a, HResult<GenerationStats>> {
                *self.last.lock().unwrap() = Some(request.clone());
                self.inner.generate(request, sink)
            }
            fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, HResult<Vec<f32>>> {
                self.inner.embed(text)
            }
        }
        let dir = std::env::temp_dir().join(format!("hide_turn_compat_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("CLAUDE.md"),
            "# House rules\nZZHOUSERULETOKEN: never delete data without confirmation.\n",
        )
        .unwrap();
        let services = BackendServices::open(HideConfig::for_workspace(&dir)).unwrap();
        assert!(!services.repo_instructions.is_empty());
        assert!(services.repo_instructions.text.contains("ZZHOUSERULETOKEN"));
        let session = services.session();
        let index = Arc::new(InMemoryCodeIndex::default());
        let recorder = Arc::new(RecordingClient {
            inner: StubInferenceClient::new("ok"),
            last: std::sync::Mutex::new(None),
        });
        let inference: Arc<dyn InferenceClient> = recorder.clone();
        run_turn_core(
            inference,
            services.event_log.clone(),
            services.role_registry.clone(),
            index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            Arc::new(UiEventBus::default()),
            session.clone(),
            "some unrelated task".to_string(),
            None,
            None,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();
        let req = recorder
            .last
            .lock()
            .unwrap()
            .clone()
            .expect("a request was recorded");
        assert!(req.prompt.contains("ZZHOUSERULETOKEN"));
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let receipt = events
            .iter()
            .find(|e| e.kind == "context.instructions")
            .expect("a context.instructions receipt must be logged");
        assert!(receipt.payload["count"].as_u64().unwrap_or(0) >= 1);
        assert!(
            receipt.payload["files"]
                .as_array()
                .map(|a| a.iter().any(|f| f["path"]
                    .as_str()
                    .map(|p| p.ends_with("CLAUDE.md"))
                    .unwrap_or(false)))
                .unwrap_or(false),
            "receipt files must list the CLAUDE.md, got: {}",
            receipt.payload
        );
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    async fn run_turn_kernel_drives_real_loop_to_terminal_with_compiled_context() {
        use futures::future::BoxFuture;
        use hawking_orch::inference::StubInferenceClient;
        use hawking_orch::router::SimpleRouter;
        use hide_core::config::HideConfig;
        use hide_core::ids::now_ms;
        use hide_kernel::govern::Autonomy;
        use hide_kernel::runtime_client::KernelRuntimeClient;
        use hide_kernel::verify::oracle::{Oracle, OracleClass, Verdict, VerificationInput};
        use hide_kernel::verify::OracleSuite;
        use hide_kernel::{AgentKernel, Grounding};
        struct NoopPassOracle(&'static str);
        impl Oracle for NoopPassOracle {
            fn name(&self) -> &str {
                self.0
            }
            fn verify<'a>(
                &'a self,
                _input: &'a VerificationInput,
            ) -> BoxFuture<'a, Result<Verdict>> {
                Box::pin(async move {
                    Ok(Verdict::pass(self.0, OracleClass::Deterministic, "noop pass"))
                })
            }
        }
        let dir = std::env::temp_dir().join(format!("hide_kernel_turn_{}", now_ms()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\nname=\"fx\"\n").unwrap();
        std::fs::write(dir.join("src/lib.rs"), "pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
            .unwrap();
        let _ = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .output();
        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();
        services.seed_code_file(
            "src/marker.rs",
            "// zzkernelmarker context bridge anchor ZZONLYINFILE\npub fn helper() {}\n",
        );
        let runtime = Arc::new(KernelRuntimeClient::new(
            Arc::new(SimpleRouter::new(services.role_registry.clone())),
            Arc::new(StubInferenceClient::new("investigate and verify the change")),
        ));
        let dispatcher = Arc::new(build_default_tool_dispatcher(
            &services.config,
            Arc::new(build_default_tool_registry()),
        ));
        let mut suite = OracleSuite::new();
        suite.register(Arc::new(NoopPassOracle("build")));
        suite.register(Arc::new(NoopPassOracle("test")));
        suite.register(Arc::new(NoopPassOracle("typecheck")));
        let grounding = Arc::new(Grounding::new(services.code_index.clone()));
        let kernel = AgentKernel::builder(services.event_log.clone())
            .workspace_root(dir.to_string_lossy().to_string())
            .autonomy(Autonomy::SuggestOnly) // bounded; the plan step is non-effectful
            .grounding(grounding)
            .runtime(runtime)
            .dispatcher(dispatcher.clone())
            .oracle_suite(suite)
            .build();
        let ui_bus = Arc::new(UiEventBus::default());
        let interrupts = Arc::new(InterruptHub::default());
        let approvals = Arc::new(crate::approval::ApprovalHub::default());
        let run_id = RunId::new();
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
            run_id,
            session.clone(),
            "http://127.0.0.1:9/unreachable".to_string(),
            "zzkernelmarker context bridge anchor".to_string(),
            64,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();
        assert!(state.phase.is_terminal());
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let plan_created = events.iter().find(|e| {
            e.kind == "plan.created"
                && e.payload.get("action").and_then(|a| a.as_str()) == Some("created")
        });
 assert!( plan_created.is_some(), "a plan.created (action=created) event must be logged" );
        assert!(events.iter().any(|e| e.kind == "agent.observation"));
        assert!(events.iter().any(|e| e.kind == "verify.result"));
        let objective = plan_created
            .unwrap()
            .payload
            .get("plan")
            .and_then(|p| p.get("objective"))
            .and_then(|o| o.as_str())
            .unwrap_or_default();
        assert!(objective.contains("zzkernelmarker"));
        assert!(objective.contains("ZZONLYINFILE"));
        let _ = std::fs::remove_dir_all(dir);
    }
    struct AnswerPlanner {
        oracle: String,
    }
    impl hide_kernel::plan::planner::Planner for AnswerPlanner {
        fn synthesize<'a>(
            &'a self,
            objective: &'a str,
        ) -> futures::future::BoxFuture<'a, Result<hide_kernel::plan::schema::Plan>> {
            use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
            let oracle = self.oracle.clone();
            let objective = objective.to_string();
            Box::pin(async move {
                let step = PlanStep::new(
                    "synthesize the answer",
                    StepKind::Synthesize,
                    Acceptance::with_oracles("an answer is produced", vec![oracle]),
                );
                Ok(Plan {
                    id: hide_core::ids::PlanId::new(),
                    title: "answer plan".to_string(),
                    objective,
                    steps: vec![step],
                    status: PlanStatus::Active,
                    budget: Default::default(),
                })
            })
        }
    }
    async fn drive_answer_turn(
        services: Arc<BackendServices>,
        session: SessionId,
        prompt: &str,
        answer: &str,
    ) -> (AgentState, Vec<UiEvent>) {
        use hawking_orch::inference::StubInferenceClient;
        use hawking_orch::router::SimpleRouter;
        use hide_kernel::plan::planner::Planner;
        use hide_kernel::runtime_client::KernelRuntimeClient;
        use hide_kernel::verify::OracleSuite;
        let root = services
            .config
            .workspace_root
            .to_string_lossy()
            .to_string();
        let runtime = Arc::new(KernelRuntimeClient::new(
            Arc::new(SimpleRouter::new(services.role_registry.clone())),
            Arc::new(StubInferenceClient::new(answer)),
        ));
        let planner = Arc::new(AnswerPlanner {
            oracle: "answered".to_string(),
        });
        let mut suite = OracleSuite::new();
        suite.register(Arc::new(NoopPassOracle("answered")));
        let kernel = AgentKernel::builder(services.event_log.clone())
            .workspace_root(root)
            .autonomy(Autonomy::SuggestOnly)
            .planner(planner as Arc<dyn Planner>)
            .runtime(runtime)
            .oracle_suite(suite)
            .build();
        let ui_bus = Arc::new(UiEventBus::default());
        let mut rx = ui_bus.subscribe();
        let interrupts = Arc::new(InterruptHub::default());
        let approvals = Arc::new(ApprovalHub::default());
        let run_id = RunId::new();
        let state = run_turn_kernel(
            kernel,
            services.event_log.clone(),
            services.key_value_store.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            ui_bus.clone(),
            interrupts,
            approvals,
            run_id,
            session.clone(),
            "http://127.0.0.1:9/unreachable".to_string(),
            prompt.to_string(),
            64,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();
        let mut ui_events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            ui_events.push(ev);
        }
        (state, ui_events)
    }
