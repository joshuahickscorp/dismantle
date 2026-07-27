//! One trait/ABI for architecture-family adapters.

use crate::support_level::SupportLevel;

/// A named piece of evidence that backs a support level.
///
/// Registry tests require that every non-`DECLARED` level has at least one
/// evidence entry whose `path` exists relative to the workspace root (or is an
/// explicitly allowed receipt name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Evidence {
    /// Repo-relative path to a test, receipt JSON, or module.
    pub path: &'static str,
    /// One-line claim this evidence supports.
    pub claim: &'static str,
}

/// Static descriptor exported by each family module.
#[derive(Debug, Clone, Copy)]
pub struct FamilyDescriptor {
    pub id: &'static str,
    pub display_name: &'static str,
    pub level: SupportLevel,
    pub evidence: &'static [Evidence],
    /// Rust module path (or pack id) that implements / declares the family.
    pub module: &'static str,
    /// True when *some* path in-tree can execute a forward/generate for this family.
    pub executes: bool,
    /// True when `load_engine` (or gravity registry) dispatches this family today.
    pub serve_registered: bool,
    pub gaps: &'static [&'static str],
}

/// The one family-adapter ABI. Implementations are thin static descriptors;
/// this trait exists so the registry can be extended without a giant match.
pub trait FamilyAdapter: Send + Sync {
    fn descriptor(&self) -> FamilyDescriptor;

    fn family_id(&self) -> &'static str {
        self.descriptor().id
    }

    fn support_level(&self) -> SupportLevel {
        self.descriptor().level
    }
}

/// Blanket helper for unit-struct family modules.
pub fn describe(
    id: &'static str,
    display_name: &'static str,
    level: SupportLevel,
    evidence: &'static [Evidence],
    module: &'static str,
    executes: bool,
    serve_registered: bool,
    gaps: &'static [&'static str],
) -> FamilyDescriptor {
    FamilyDescriptor {
        id,
        display_name,
        level,
        evidence,
        module,
        executes,
        serve_registered,
        gaps,
    }
}
