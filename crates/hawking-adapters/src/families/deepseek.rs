//! DeepSeek V2 family (GGUF deepseek2).

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct DeepSeekFamily;

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/deepseek_v2.rs",
        claim: "in-tree DeepSeekV2 engine",
    },
    Evidence {
        path: "crates/hawking-core/tests/cpu_backend_parity_deepseek.rs",
        claim: "CPU backend parity for deepseek path",
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine dispatches deepseek2",
    },
];

const GAPS: &[&str] = &[
    "not FULL_PARENT_VALIDATED: no sealed full-size parent receipt in registry evidence",
    "not PRODUCTION",
];

impl FamilyAdapter for DeepSeekFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "deepseek",
            "DeepSeek V2",
            SupportLevel::SmallRealCheckpoint,
            EVIDENCE,
            "crates/hawking-core/src/model/deepseek_v2.rs",
            true,
            true,
            GAPS,
        )
    }
}
