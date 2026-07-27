//! Honest support levels. Never inflate.

use serde::{Deserialize, Serialize};

/// Ladder of evidence for a model family. Literal meanings; do not promote
/// from a code reading alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportLevel {
    /// The family is described; nothing executes.
    Declared,
    /// Matches a reference implementation on synthetic tensors.
    SyntheticParity,
    /// Matches on a real small checkpoint of the family.
    SmallRealCheckpoint,
    /// Matches on a real full-size parent.
    FullParentValidated,
    /// Served, under test, with a standing parity receipt.
    /// **No family is at this level today.**
    Production,
}

impl SupportLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            SupportLevel::Declared => "DECLARED",
            SupportLevel::SyntheticParity => "SYNTHETIC_PARITY",
            SupportLevel::SmallRealCheckpoint => "SMALL_REAL_CHECKPOINT",
            SupportLevel::FullParentValidated => "FULL_PARENT_VALIDATED",
            SupportLevel::Production => "PRODUCTION",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "DECLARED" => Some(SupportLevel::Declared),
            "SYNTHETIC_PARITY" => Some(SupportLevel::SyntheticParity),
            "SMALL_REAL_CHECKPOINT" => Some(SupportLevel::SmallRealCheckpoint),
            "FULL_PARENT_VALIDATED" => Some(SupportLevel::FullParentValidated),
            "PRODUCTION" => Some(SupportLevel::Production),
            _ => None,
        }
    }
}
