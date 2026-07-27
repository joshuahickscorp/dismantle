//! GLM family (flagship via .gravity / glm_moe_dsa).
//!
//! Survey claim (re-verified): SMALL_REAL_CHECKPOINT with M04_SEALED receipt
//! in GLM52_FLAGSHIP_ADAPTER_PARITY.json. Not PRODUCTION.

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct GlmFamily;

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "GLM52_FLAGSHIP_ADAPTER_PARITY.json",
        claim: "M04_SEALED: Rust adapter vs oracle on real flagship .gravity shards",
    },
    Evidence {
        path: "crates/hawking-core/src/model/gravity_engine.rs",
        claim: "GravityEngine dispatches glm_moe_dsa",
    },
    Evidence {
        path: "crates/hawking-core/tests/gravity_engine_registry.rs",
        claim: "registry path for .gravity artifacts",
    },
];

const GAPS: &[&str] = &[
    "not PRODUCTION",
    "gravity_glm.rs is another lane's sealed path — not claimed as open production serve",
    "full parent source safetensors not the parity authority (gravity bytes are)",
];

impl FamilyAdapter for GlmFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "glm",
            "GLM (gravity glm_moe_dsa)",
            SupportLevel::SmallRealCheckpoint,
            EVIDENCE,
            "crates/hawking-core/src/model/gravity_engine.rs",
            true,
            true, // .gravity path is registered in load_engine
            GAPS,
        )
    }
}
