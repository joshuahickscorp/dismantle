//! State-space family (RWKV7 in-tree; Mamba2 extracted).

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct StateSpaceFamily;

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/rwkv7.rs",
        claim: "in-tree RwkvSeven engine",
    },
    Evidence {
        path: "crates/hawking-core/tests/rwkv7_parity.rs",
        claim: "RWKV7 parity + load_engine routing",
    },
    Evidence {
        path: "packs/hawking-adapters-extra.json",
        claim: "mamba2 extracted off-tree",
    },
];

const GAPS: &[&str] = &[
    "mamba2 not in shipping load_engine",
    "not PRODUCTION",
    "family spans RWKV (executes) and Mamba2 (declared pack only)",
];

impl FamilyAdapter for StateSpaceFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "state_space",
            "State-space (RWKV7 + Mamba2)",
            // RWKV has real checkpoint parity tests; mamba2 does not ship.
            SupportLevel::SmallRealCheckpoint,
            EVIDENCE,
            "crates/hawking-core/src/model/rwkv7.rs",
            true, // rwkv7 executes
            true, // rwkv7 serve-registered
            GAPS,
        )
    }
}
