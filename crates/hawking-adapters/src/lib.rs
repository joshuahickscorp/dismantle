//! # One model-family adapter ABI + honest support-level registry
//!
//! This is **not** the LoRA selection registry in `hawking-orch::adapters`
//! (language/task LoRAs). This is the **architecture-family** registry that
//! answers: for Llama / Qwen / GLM / …, what is the true support level, does
//! anything execute, and is the family serve-registered?
//!
//! ## Support levels (never inflate)
//!
//! ```text
//! DECLARED               family described; nothing executes
//! SYNTHETIC_PARITY       matches reference on synthetic tensors
//! SMALL_REAL_CHECKPOINT  matches on a real small checkpoint of the family
//! FULL_PARENT_VALIDATED  matches on a real full-size parent
//! PRODUCTION             served, under test, with a standing parity receipt
//! ```
//!
//! **No family is PRODUCTION today.** Promoting a level requires a named
//! artifact or test in [`Evidence`], enforced by the registry test.
//!
//! ## Layout
//!
//! One module per family under [`families`]; [`registry`] indexes them.
//! [`generate`] emits docs, schemas, CLI validation, SDK types, HIDE
//! capability declarations, and Fabric declarations — same deterministic
//! golden-file pattern as `hide-sdk-codegen`.

pub mod abi;
pub mod evidence;
pub mod export;
pub mod families;
pub mod generate;
pub mod registry;
pub mod support_level;

pub use abi::{Evidence, FamilyAdapter, FamilyDescriptor};
pub use export::{adapter_registry_document, adapter_registry_json};
pub use generate::{generate_all, GeneratedArtifact};
pub use registry::{builtin_registry, FamilyRegistry};
pub use support_level::SupportLevel;

/// Schema id for `HAWKING_ADAPTER_REGISTRY.json`.
pub const REGISTRY_SCHEMA: &str = "hawking.adapters.registry.v1";
