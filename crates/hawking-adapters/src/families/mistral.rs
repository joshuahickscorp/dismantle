//! Mistral / Mixtral family.
//!
//! Dense Mistral GGUF shares the llama loader (`load_engine` maps `mistral` →
//! LlamaDense). Mixtral MoE was extracted to the hawking-adapters-extra pack.

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct MistralFamily;

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "dense mistral arch string routes to LlamaDense",
    },
    Evidence {
        path: "packs/hawking-adapters-extra.json",
        claim: "mixtral extracted off-tree to adapters-extra pack",
    },
    Evidence {
        path: "crates/hawking-seed-c/src/providers/adapters.rs",
        claim: "seed-c ArchAdapter::mixtral is declarative plan-only (does not execute)",
    },
];

const GAPS: &[&str] = &[
    "mixtral MoE not in shipping load_engine",
    "seed-c ArchAdapter does not execute",
    "no PRODUCTION receipt",
];

impl FamilyAdapter for MistralFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "mistral_mixtral",
            "Mistral / Mixtral",
            // Dense mistral can execute via llama loader; mixtral does not.
            // Overall family honesty: SMALL_REAL for dense path only is too
            // strong for the MoE half — stay at SMALL_REAL_CHECKPOINT for the
            // dense mistral route that shares llama evidence, with gaps named.
            SupportLevel::SmallRealCheckpoint,
            EVIDENCE,
            "crates/hawking-core/src/model/llama.rs (+ pack mixtral)",
            true, // dense mistral executes
            true, // dense mistral arch is serve-registered
            GAPS,
        )
    }
}
