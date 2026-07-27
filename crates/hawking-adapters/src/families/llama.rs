//! Llama family (Llama-2/3.x dense via GGUF + gravity llama).

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct LlamaFamily;

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/llama.rs",
        claim: "in-tree LlamaDense engine module",
    },
    Evidence {
        path: "crates/hawking-core/tests/llama32_smoke.rs",
        claim: "small-parent greedy smoke when GGUF present",
    },
    Evidence {
        path: "crates/hawking-core/tests/gravity_llama_forward.rs",
        claim: "gravity llama forward path",
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine dispatches llama|mistral GGUF arch strings",
    },
];

const GAPS: &[&str] = &[
    "no standing PRODUCTION parity receipt",
    "smoke tests skip when weights absent",
];

impl FamilyAdapter for LlamaFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "llama",
            "Llama (dense GGUF + gravity)",
            // Real small checkpoints exercise the path; not claiming full parent
            // validation without a sealed full-size parent receipt.
            SupportLevel::SmallRealCheckpoint,
            EVIDENCE,
            "crates/hawking-core/src/model/llama.rs",
            true,
            true,
            GAPS,
        )
    }
}
