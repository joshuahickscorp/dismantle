//! Gemma family — extracted to hawking-adapters-extra pack.

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct GemmaFamily;

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "packs/hawking-adapters-extra.json",
        claim: "gemma2 extracted off-tree",
    },
    Evidence {
        path: "crates/hawking-core/tests/gemma2_smoke.rs",
        claim: "smoke test remains but load_engine rejects unknown gemma2 arch without pack",
    },
    Evidence {
        path: "crates/hawking-seed-c/src/providers/adapters.rs",
        claim: "seed-c ArchAdapter::gemma2 is declarative plan-only",
    },
];

const GAPS: &[&str] = &[
    "module not in shipping load_engine",
    "pack hydrate required to execute",
    "not PRODUCTION",
];

impl FamilyAdapter for GemmaFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "gemma",
            "Gemma 2",
            SupportLevel::Declared,
            EVIDENCE,
            "packs/hawking-adapters-extra (gemma2)",
            false,
            false,
            GAPS,
        )
    }
}
