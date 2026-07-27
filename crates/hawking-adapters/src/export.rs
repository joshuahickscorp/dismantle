//! Export `HAWKING_ADAPTER_REGISTRY.json` (schema `hawking.adapters.registry.v1`).

use serde_json::{json, Value};

use crate::registry::builtin_registry;
use crate::REGISTRY_SCHEMA;

pub fn adapter_registry_document() -> Value {
    let r = builtin_registry();
    let mut families = Vec::new();
    for d in r.families() {
        families.push(json!({
            "id": d.id,
            "display_name": d.display_name,
            "level": d.level.as_str(),
            "evidence": d.evidence.iter().map(|e| json!({
                "path": e.path,
                "claim": e.claim,
            })).collect::<Vec<_>>(),
            "module": d.module,
            "executes": d.executes,
            "serve_registered": d.serve_registered,
            "gaps": d.gaps,
        }));
    }

    json!({
        "schema": REGISTRY_SCHEMA,
        "note": "No family is PRODUCTION today. Levels are never inflated from a code reading alone.",
        "support_levels": [
            "DECLARED",
            "SYNTHETIC_PARITY",
            "SMALL_REAL_CHECKPOINT",
            "FULL_PARENT_VALIDATED",
            "PRODUCTION"
        ],
        "authorities_not_sole": [
            {
                "name": "load_engine",
                "path": "crates/hawking-core/src/model/mod.rs",
                "role": "GGUF live dispatch (llama, deepseek2, qwen, qwen-moe, rwkv7) + gravity"
            },
            {
                "name": "gravity_engine",
                "path": "crates/hawking-core/src/model/gravity_engine.rs",
                "role": "live .gravity for llama + glm_moe_dsa only"
            },
            {
                "name": "seed-c ArchAdapter",
                "path": "crates/hawking-seed-c/src/providers/adapters.rs",
                "role": "declarative plan summary — does not execute"
            },
            {
                "name": "PRODUCTION_EXECUTION_ADAPTER_REGISTRY",
                "path": "tools/condense/glm52_worker.py",
                "role": "empty by contract (fail-closed)"
            },
            {
                "name": "hawking-adapters-extra",
                "path": "packs/hawking-adapters-extra.json",
                "role": "gemma2/phi3/mixtral/mamba2/olmoe extracted off-tree"
            },
            {
                "name": "hawking-adapters FamilyRegistry",
                "path": "crates/hawking-adapters/src/registry.rs",
                "role": "THIS crate — honest support-level index (metadata ABI, not a second runtime)"
            }
        ],
        "families": families,
    })
}

pub fn adapter_registry_json() -> String {
    let mut s = serde_json::to_string_pretty(&adapter_registry_document())
        .expect("registry document serializes");
    s.push('\n');
    s
}
