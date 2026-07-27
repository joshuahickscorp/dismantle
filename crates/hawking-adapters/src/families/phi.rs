//! Phi family — extracted to hawking-adapters-extra pack.

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct PhiFamily;

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "packs/hawking-adapters-extra.json",
        claim: "phi3 extracted off-tree",
    },
    Evidence {
        path: "crates/hawking-core/tests/phi3_smoke.rs",
        claim: "smoke test remains; arch not in shipping load_engine",
    },
    Evidence {
        path: "crates/hawking-seed-c/src/providers/adapters.rs",
        claim: "seed-c ArchAdapter::phi3 is declarative plan-only",
    },
];

const GAPS: &[&str] = &[
    "module not in shipping load_engine",
    "pack hydrate required to execute",
    "not PRODUCTION",
];

impl FamilyAdapter for PhiFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "phi",
            "Phi-3",
            SupportLevel::Declared,
            EVIDENCE,
            "packs/hawking-adapters-extra (phi3)",
            false,
            false,
            GAPS,
        )
    }
}
