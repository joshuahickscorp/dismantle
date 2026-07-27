//! Codegen entry: regenerate checked-in adapter/event artifacts.
//!
//! Mirrors `hide-sdk-codegen`: pure deterministic write of goldens; drift tests
//! fail when regeneration would change bytes.
//!
//! Also writes the three repo-root deliverables when `--repo-root` is set
//! (default: two levels above this crate).

use std::path::PathBuf;

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let written = hawking_adapters::generate::write_all(&crate_root)
        .context("write generated/ under hawking-adapters")?;
    for p in &written {
        println!("wrote {} ({} bytes)", p.display(), std::fs::metadata(p)?.len());
    }

    // Repo-root deliverables (contract).
    let repo_root = crate_root
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .context("resolve repo root")?;

    let root_artifacts = [
        (
            "HAWKING_ADAPTER_REGISTRY.json",
            hawking_adapters::adapter_registry_json(),
        ),
        (
            "HAWKING_CANONICAL_EVENTS.json",
            hawking_events::canonical_events_json(),
        ),
        (
            "HAWKING_BRIDGE_SURFACE.json",
            // Bridge surface is owned by hawking-serve; we re-export a
            // static declaration here so codegen stays one command. The
            // serve crate's module is the live source — keep in sync via
            // the bridge_surface module string below matching serve.
            bridge_surface_json(),
        ),
    ];
    for (name, contents) in &root_artifacts {
        let path = repo_root.join(name);
        std::fs::write(&path, contents).with_context(|| format!("write {}", path.display()))?;
        println!("wrote {} ({} bytes)", path.display(), contents.len());
    }
    Ok(())
}

/// Keep in lockstep with `hawking_serve::surface::bridge_surface_json`.
fn bridge_surface_json() -> String {
    // Duplicated as a string builder to avoid a hawking-serve dep from this
    // codegen bin (serve pulls Metal/core). The serve crate tests assert equality.
    let doc = serde_json::json!({
        "schema": "hawking.bridge.surface.v1",
        "endpoints": [
            {
                "endpoint": "POST /v1/chat/completions",
                "status": "live",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> chat_completions",
                "tests": ["crates/hawking-serve/tests/http_integration.rs"]
            },
            {
                "endpoint": "POST /v1/completions",
                "status": "live",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> completions",
                "tests": ["crates/hawking-serve/tests/http_integration.rs"]
            },
            {
                "endpoint": "GET /v1/models",
                "status": "live",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> list_models",
                "tests": ["crates/hawking-serve/tests/http_integration.rs"]
            },
            {
                "endpoint": "GET /healthz",
                "status": "live",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> healthz",
                "tests": ["crates/hawking-serve/tests/http_integration.rs"]
            },
            {
                "endpoint": "GET /metrics",
                "status": "live",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> metrics",
                "tests": []
            },
            {
                "endpoint": "POST /v1/embeddings",
                "status": "partial",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> embeddings",
                "tests": []
            },
            {
                "endpoint": "POST /v1/hawking/tokens",
                "status": "live",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> hawking_tokens",
                "tests": ["crates/hawking-serve/tests/hawking_native_endpoint.rs"]
            },
            {
                "endpoint": "POST /v1/hawking/generate",
                "status": "live",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> hawking_generate",
                "tests": ["crates/hawking-serve/tests/hawking_native_endpoint.rs"]
            },
            {
                "endpoint": "GET /v1/hawking/context",
                "status": "live",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> hawking_context",
                "tests": []
            },
            {
                "endpoint": "GET /v1/hawking/surface",
                "status": "live",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> hawking_surface",
                "tests": ["crates/hawking-serve/tests/http_integration.rs"]
            },
            {
                "endpoint": "POST /v1/responses",
                "status": "not_implemented",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> not_implemented_responses",
                "tests": ["crates/hawking-serve/tests/http_integration.rs"]
            },
            {
                "endpoint": "POST /v1/messages",
                "status": "not_implemented",
                "entry_path": "crates/hawking-serve/src/http.rs:router -> not_implemented_anthropic_messages",
                "tests": ["crates/hawking-serve/tests/http_integration.rs"]
            },
            {
                "endpoint": "MCP",
                "status": "partial",
                "entry_path": "crates/hide-backend (register_mcp_servers_at_boot on hide tree)",
                "tests": []
            },
            {
                "endpoint": "ACP",
                "status": "partial",
                "entry_path": "crates/hide-acp (DeferredTurnHandler / capability negotiate)",
                "tests": []
            },
            {
                "endpoint": "SDK Transport -> hide-serve",
                "status": "not_implemented",
                "entry_path": "crates/hide-sdk/src/client.rs (MockTransport only; real transport deferred)",
                "tests": ["crates/hide-sdk/tests/client.rs"]
            }
        ]
    });
    let mut s = serde_json::to_string_pretty(&doc).unwrap();
    s.push('\n');
    s
}
