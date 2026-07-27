//! Kimi family.
//!
//! Survey claim (re-verified): reference / synthetic only, not serve-registered.
//! KIMI_K26_ADAPTER_TWIN.json runtime_claim = SYNTHETIC_CPU_REFERENCE_AND_BOUND_REAL_SOURCE_METAL_K1.

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct KimiFamily;

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "KIMI_K26_ADAPTER_TWIN.json",
        claim: "synthetic CPU reference + bound real-source metal K1 twin",
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine has no kimi arch arm (not serve-registered)",
    },
];

const GAPS: &[&str] = &[
    "not serve-registered in load_engine",
    "no SMALL_REAL_CHECKPOINT sealed receipt for full generate path",
    "not PRODUCTION",
];

impl FamilyAdapter for KimiFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "kimi",
            "Kimi K2.x",
            SupportLevel::SyntheticParity,
            EVIDENCE,
            "KIMI_K26_ADAPTER_TWIN.json (reference twin; no in-tree serve module)",
            false,
            false,
            GAPS,
        )
    }
}
