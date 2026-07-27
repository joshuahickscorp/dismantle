//! MiniMax family — declared only; no in-tree execution path.

use crate::abi::{describe, Evidence, FamilyAdapter, FamilyDescriptor};
use crate::support_level::SupportLevel;

pub struct MiniMaxFamily;

const EVIDENCE: &[Evidence] = &[Evidence {
    path: "FABRIC_BRIDGE_ARCHAEOLOGY.md",
    claim: "family listed in bridge archaeology; no serve path found",
}];

const GAPS: &[&str] = &[
    "no in-tree engine module",
    "not serve-registered",
    "not PRODUCTION",
];

impl FamilyAdapter for MiniMaxFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        describe(
            "minimax",
            "MiniMax",
            SupportLevel::Declared,
            EVIDENCE,
            "(none — declared only)",
            false,
            false,
            GAPS,
        )
    }
}
