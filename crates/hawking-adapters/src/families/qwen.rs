//! Qwen dense / MoE family.
//!
//! Survey claim (re-verified): FULL_PARENT_VALIDATED on small parents.
//! serve-registered via load_engine. Not PRODUCTION (no standing production receipt).

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct QwenFamily;

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/qwen_dense.rs",
        claim: "in-tree QwenDense engine",
    },
    Evidence {
        path: "crates/hawking-core/src/model/qwen_moe.rs",
        claim: "in-tree QwenMoE engine",
    },
    Evidence {
        path: "crates/hawking-core/tests/integration_greedy_64.rs",
        claim: "greedy integration path used as parent-validation gate",
    },
    Evidence {
        path: "crates/hawking-core/tests/cpu_backend_parity.rs",
        claim: "CPU backend parity against live engine",
    },
    Evidence {
        path: "crates/hawking-core/tests/qwen_tq_serve_parity.rs",
        claim: "TQ serve parity test (feature-gated)",
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine dispatches qwen2/qwen2moe",
    },
];

const GAPS: &[&str] = &[
    "not PRODUCTION: no standing production parity receipt under continuous serve",
    "large MoE parents (235B/397B) are campaign-side, not this registry's PRODUCTION claim",
];

impl FamilyAdapter for QwenFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "qwen",
            "Qwen (dense + MoE)",
            SupportLevel::FullParentValidated,
            EVIDENCE,
            "crates/hawking-core/src/model/qwen_dense.rs",
            true,
            true,
            GAPS,
        )
    }
}
