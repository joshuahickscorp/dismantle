//! hide-state: state-capsule schemas, integrity, and compatibility binding.
//!
//! # Superseded as the live checkpoint authority
//!
//! **This crate is NOT wired into the host.** Durable session restore boundaries
//! live under `hide-backend`'s [`CheckpointStore`] / `BackendHost::checkpoint_*`
//! (event-log KV). RPC `state/save|load|fork|release` routes onto that host
//! implementation (see `hide-backend::rpc`), not onto the capsule store below.
//! Wiring a second checkpoint authority would violate the consolidation law of
//! one session/event authority. Keep this crate only as schema reference /
//! offline fixtures; do not depend on it from `hide-backend` or `hide-serve`.
//!
//! HIDE carries agent state as capsules: opaque runtime bytes wrapped in a
//! header that identifies them and a digest that lets a reader prove they are
//! intact (Bible sec 23, sec 56). This crate defines those schemas and the
//! deterministic logic around them: what a capsule is, how it serializes to a
//! self-describing byte stream, how a reader verifies it, when it is allowed to
//! bind to a runtime, and how a store saves, loads, forks, compares, releases,
//! and inspects capsules.
//!
//! Scope: this crate is schema-only and entirely model-free. It describes and
//! verifies capsule bytes over synthetic fixtures. It never runs a model, never
//! produces or consumes live runtime state, and makes no assertion about
//! runtime performance or output quality (Bible law 17, sec 56 gate). The
//! runtime that actually produces these bytes from a live engine and rebinds
//! them is DEFERRED_MODEL_REQUIRED: it connects later, against these schemas,
//! and nothing here should be read as a claim that it exists yet.
//!
//! The invariants this crate holds:
//!
//! - Self-describing bytes. A serialized capsule carries its own magic tag,
//!   format version, metadata, and integrity digest, so a reader needs no
//!   out-of-band convention to parse and check it.
//!
//! - Integrity on every load. [`Capsule::from_bytes`] recomputes the payload
//!   digest and rejects any stream whose payload was altered.
//!
//! - Honest ancestry. [`Capsule::fork`] copies the payload byte for byte under
//!   a fresh id and records the parent, so lineage is always recoverable.
//!
//! - Strict binding. A capsule refuses to load into a runtime whose identity
//!   disagrees on any field, and says exactly which field via a typed
//!   [`IncompatibleReason`] rather than a scraped string.
//!
//! ```
//! use crate::state::{
//!     CapsuleBuilder, CapsuleType, CapsuleStore, IdentityBinding, MemoryStore,
//! };
//!
//! let identity = IdentityBinding {
//!     model_weights_id: "w".into(),
//!     arch_id: "a".into(),
//!     tokenizer_id: "t".into(),
//!     prompt_abi_version: "1".into(),
//!     tool_registry_id: "r".into(),
//!     engine_build_id: "b".into(),
//!     security_domain: "d".into(),
//! };
//!
//! let capsule = CapsuleBuilder::new(CapsuleType::Recurrent, "model-x", identity.clone())
//!     .runtime_version("rt-1")
//!     .seal(vec![1, 2, 3, 4]);
//!
//! let mut store = MemoryStore::new();
//! let id = store.save(&capsule).unwrap();
//! let loaded = store.load(&id).unwrap();
//! assert_eq!(loaded.payload(), &[1, 2, 3, 4]);
//! assert!(loaded.is_loadable(&identity).is_ok());
//! ```

pub use capsule::{Capsule, CapsuleBuilder, CapsuleInspect};
pub use error::{CapsuleError, IncompatibleReason, Result};
pub use header::{now_ms, CapsuleHeader, CapsuleId, CapsuleType};
pub use identity::IdentityBinding;
pub use integrity::{Integrity, IntegrityAlgo};
pub use store::{Ancestry, CapsuleComparison, CapsuleStore, DiskStore, MemoryStore};

// --- inlined state/capsule.rs ---
pub mod capsule {
//! The capsule itself: header, identity, payload, and its byte format.
//!
//! A capsule serializes to a self-describing byte stream: a magic tag, a format
//! version, a length-prefixed JSON metadata block (header plus identity), then
//! the raw payload. Reading is integrity-checked: [`Capsule::from_bytes`]
//! rejects a stream whose payload length or digest disagrees with the header,
//! so a flipped byte can never be loaded silently.

use serde::{Deserialize, Serialize};

use crate::state::error::{CapsuleError, IncompatibleReason, Result};
use crate::state::header::{now_ms, CapsuleHeader, CapsuleId, CapsuleType};
use crate::state::identity::IdentityBinding;
use crate::state::integrity::{Integrity, IntegrityAlgo};

/// Magic tag at the head of every serialized capsule.
const MAGIC: &[u8; 8] = b"HIDECAP1";
/// The byte format this build writes and reads.
const FORMAT_VERSION: u16 = 1;
/// Fixed-size prefix: magic (8) plus version (2) plus meta length (4).
const PREFIX_LEN: usize = 8 + 2 + 4;

/// Owned metadata block, used when reading a capsule back from bytes.
#[derive(Deserialize)]
struct MetaOwned {
    header: CapsuleHeader,
    identity: IdentityBinding,
}

/// Borrowing metadata block, used when writing a capsule so serialization does
/// not clone the header and identity.
#[derive(Serialize)]
struct MetaRef<'a> {
    header: &'a CapsuleHeader,
    identity: &'a IdentityBinding,
}

/// Capsule metadata without the payload, returned by [`Capsule::inspect`].
///
/// Everything needed to identify and audit a capsule is here; the payload is
/// deliberately absent so an inspector never has to materialize it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleInspect {
    pub header: CapsuleHeader,
    pub identity: IdentityBinding,
}

/// A sealed capsule: descriptive header, identity binding, and opaque payload.
///
/// The payload is private so a capsule cannot drift out of agreement with the
/// length and digest recorded in its header; read it with [`Capsule::payload`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capsule {
    header: CapsuleHeader,
    identity: IdentityBinding,
    payload: Vec<u8>,
}

impl Capsule {
    pub fn header(&self) -> &CapsuleHeader {
        &self.header
    }

    pub fn identity(&self) -> &IdentityBinding {
        &self.identity
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn capsule_id(&self) -> &CapsuleId {
        &self.header.capsule_id
    }

    pub fn parent_capsule_id(&self) -> Option<&CapsuleId> {
        self.header.parent_capsule_id.as_ref()
    }

    /// Whether this capsule can bind to a live runtime identity. Delegates to
    /// [`IdentityBinding::is_loadable`].
    pub fn is_loadable(&self, live: &IdentityBinding) -> std::result::Result<(), IncompatibleReason> {
        self.identity.is_loadable(live)
    }

    /// Fork this capsule: a byte-for-byte copy of the payload under a fresh id,
    /// with ancestry recorded. The new capsule's `parent_capsule_id` points at
    /// this capsule and its `created_at` is refreshed; everything else,
    /// including the payload bytes and integrity digest, is preserved.
    pub fn fork(&self) -> Capsule {
        let mut header = self.header.clone();
        header.parent_capsule_id = Some(self.header.capsule_id.clone());
        header.capsule_id = CapsuleId::new();
        header.created_at = now_ms();
        Capsule {
            header,
            identity: self.identity.clone(),
            payload: self.payload.clone(),
        }
    }

    /// Release the capsule's payload, zeroing the bytes before dropping them,
    /// and return the number of bytes reclaimed. Consumes the capsule.
    pub fn release(mut self) -> usize {
        let n = self.payload.len();
        for byte in self.payload.iter_mut() {
            *byte = 0;
        }
        self.payload.clear();
        n
    }

    /// Metadata without the payload.
    pub fn inspect(&self) -> CapsuleInspect {
        CapsuleInspect {
            header: self.header.clone(),
            identity: self.identity.clone(),
        }
    }

    /// Serialize to the self-describing, integrity-carrying byte stream.
    pub fn to_bytes(&self) -> Vec<u8> {
        let meta = MetaRef {
            header: &self.header,
            identity: &self.identity,
        };
        // Header and identity are plain data with infallible serialization.
        let meta_json = serde_json::to_vec(&meta).expect("capsule metadata serializes");
        let mut out = Vec::with_capacity(PREFIX_LEN + meta_json.len() + self.payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&meta_json);
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parse a capsule from bytes, verifying the payload length and digest
    /// against the header. Rejects a truncated stream, a bad magic tag, an
    /// unknown version, a length mismatch, or a digest mismatch.
    pub fn from_bytes(bytes: &[u8]) -> Result<Capsule> {
        let (meta, meta_end) = parse_prefix_and_meta(bytes)?;
        let payload = bytes[meta_end..].to_vec();

        if payload.len() as u64 != meta.header.bytes {
            return Err(CapsuleError::LengthMismatch {
                declared: meta.header.bytes,
                actual: payload.len() as u64,
            });
        }
        if !meta.header.integrity.verify(&payload) {
            return Err(CapsuleError::IntegrityMismatch);
        }
        Ok(Capsule {
            header: meta.header,
            identity: meta.identity,
            payload,
        })
    }

    /// Parse only the metadata of a serialized capsule, without copying the
    /// payload. Used to inspect a stored capsule cheaply. The prefix and
    /// metadata are validated; the payload bytes are not read into memory, so
    /// this does not verify the payload digest.
    pub fn inspect_bytes(bytes: &[u8]) -> Result<CapsuleInspect> {
        let (meta, _meta_end) = parse_prefix_and_meta(bytes)?;
        Ok(CapsuleInspect {
            header: meta.header,
            identity: meta.identity,
        })
    }
}

/// Validate the fixed prefix and decode the metadata block. Returns the decoded
/// metadata and the offset at which the payload begins.
fn parse_prefix_and_meta(bytes: &[u8]) -> Result<(MetaOwned, usize)> {
    if bytes.len() < PREFIX_LEN {
        return Err(CapsuleError::Truncated { detail: "prefix" });
    }
    if &bytes[0..8] != MAGIC {
        return Err(CapsuleError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version != FORMAT_VERSION {
        return Err(CapsuleError::UnsupportedVersion {
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    let meta_len = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    let meta_end = PREFIX_LEN
        .checked_add(meta_len)
        .ok_or(CapsuleError::Truncated { detail: "meta length overflow" })?;
    if bytes.len() < meta_end {
        return Err(CapsuleError::Truncated { detail: "metadata" });
    }
    let meta: MetaOwned = serde_json::from_slice(&bytes[PREFIX_LEN..meta_end])?;
    Ok((meta, meta_end))
}

/// Fields common to every capsule in a sealing, gathered so [`CapsuleBuilder`]
/// stays readable. All have sensible empty defaults.
#[derive(Debug, Clone)]
pub struct CapsuleBuilder {
    capsule_type: CapsuleType,
    model_id: String,
    identity: IdentityBinding,
    model_hash: String,
    runtime_version: String,
    dtype: String,
    device: String,
    position: u64,
    context_pack_hash: String,
    parent_capsule_id: Option<CapsuleId>,
    algo: IntegrityAlgo,
}

impl CapsuleBuilder {
    /// Start a builder for a capsule of the given kind, model, and identity.
    /// The integrity algorithm defaults to blake3 and all descriptive tags
    /// default to empty; set them with the methods below before sealing.
    pub fn new(capsule_type: CapsuleType, model_id: impl Into<String>, identity: IdentityBinding) -> Self {
        CapsuleBuilder {
            capsule_type,
            model_id: model_id.into(),
            identity,
            model_hash: String::new(),
            runtime_version: String::new(),
            dtype: String::new(),
            device: String::new(),
            position: 0,
            context_pack_hash: String::new(),
            parent_capsule_id: None,
            algo: IntegrityAlgo::Blake3,
        }
    }

    pub fn model_hash(mut self, v: impl Into<String>) -> Self {
        self.model_hash = v.into();
        self
    }

    pub fn runtime_version(mut self, v: impl Into<String>) -> Self {
        self.runtime_version = v.into();
        self
    }

    pub fn dtype(mut self, v: impl Into<String>) -> Self {
        self.dtype = v.into();
        self
    }

    pub fn device(mut self, v: impl Into<String>) -> Self {
        self.device = v.into();
        self
    }

    pub fn position(mut self, v: u64) -> Self {
        self.position = v;
        self
    }

    pub fn context_pack_hash(mut self, v: impl Into<String>) -> Self {
        self.context_pack_hash = v.into();
        self
    }

    pub fn parent(mut self, v: CapsuleId) -> Self {
        self.parent_capsule_id = Some(v);
        self
    }

    pub fn integrity_algo(mut self, algo: IntegrityAlgo) -> Self {
        self.algo = algo;
        self
    }

    /// Seal the builder over `payload`, minting a fresh id and computing the
    /// payload length and integrity digest.
    pub fn seal(self, payload: Vec<u8>) -> Capsule {
        let integrity = Integrity::compute(self.algo, &payload);
        let header = CapsuleHeader {
            capsule_id: CapsuleId::new(),
            capsule_type: self.capsule_type,
            model_id: self.model_id,
            model_hash: self.model_hash,
            runtime_version: self.runtime_version,
            dtype: self.dtype,
            device: self.device,
            position: self.position,
            context_pack_hash: self.context_pack_hash,
            parent_capsule_id: self.parent_capsule_id,
            created_at: now_ms(),
            bytes: payload.len() as u64,
            integrity,
        };
        Capsule {
            header,
            identity: self.identity,
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity() -> IdentityBinding {
        IdentityBinding {
            model_weights_id: "w".into(),
            arch_id: "a".into(),
            tokenizer_id: "t".into(),
            prompt_abi_version: "1".into(),
            tool_registry_id: "r".into(),
            engine_build_id: "b".into(),
            security_domain: "d".into(),
        }
    }
    fn sample(payload: Vec<u8>) -> Capsule {
        CapsuleBuilder::new(CapsuleType::Recurrent, "model-x", identity())
            .runtime_version("rt-1")
            .dtype("f16")
            .device("metal")
            .position(42)
            .context_pack_hash("ctx-hash")
            .seal(payload)
    }
    #[test]
    fn seal_records_length_and_digest() {
        let payload = vec![9u8; 128];
        let c = sample(payload.clone());
        assert_eq!(c.header().bytes, 128);
        assert!(c.header().integrity.verify(&payload));
        assert_eq!(c.header().capsule_type, CapsuleType::Recurrent);
        assert!(c.header().parent_capsule_id.is_none());
    }
    #[test]
    fn to_bytes_from_bytes_is_byte_identical() {
        let c = sample((0u8..200).collect());
        let bytes = c.to_bytes();
        let back = Capsule::from_bytes(&bytes).unwrap();
        assert_eq!(c, back);
        assert_eq!(back.to_bytes(), bytes);
    }
    #[test]
    fn flipped_payload_byte_is_rejected() {
        let c = sample((0u8..64).collect());
        let mut bytes = c.to_bytes();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
 assert!(matches!( Capsule::from_bytes(&bytes), Err(CapsuleError::IntegrityMismatch) ));
    }
    #[test]
    fn bad_magic_and_short_stream_are_rejected() {
        assert!(matches!(
            Capsule::from_bytes(b"too short"),
            Err(CapsuleError::BadMagic) | Err(CapsuleError::Truncated { .. })
        ));
        assert!(matches!(
            Capsule::from_bytes(b"XXXXXXXX\x01\x00\x00\x00\x00\x00"),
            Err(CapsuleError::BadMagic)
        ));
    }
    #[test]
    fn wrong_version_is_rejected() {
        let c = sample(vec![1, 2, 3]);
        let mut bytes = c.to_bytes();
        bytes[8] = 0xFF; // corrupt the version low byte
 assert!(matches!( Capsule::from_bytes(&bytes), Err(CapsuleError::UnsupportedVersion { .. }) ));
    }
    #[test]
    fn fork_preserves_payload_sets_ancestry_and_new_id() {
        let parent = sample((0u8..80).collect());
        let child = parent.fork();
        assert_ne!(child.capsule_id(), parent.capsule_id());
        assert_eq!(child.parent_capsule_id(), Some(parent.capsule_id()));
        assert_eq!(child.payload(), parent.payload());
        assert_eq!(child.header().integrity, parent.header().integrity);
        assert_eq!(child.header().bytes, parent.header().bytes);
        let round = Capsule::from_bytes(&child.to_bytes()).unwrap();
        assert_eq!(round, child);
    }
    #[test]
    fn inspect_returns_metadata_without_payload() {
        let c = sample((0u8..50).collect());
        let meta = c.inspect();
        assert_eq!(meta.header, *c.header());
        assert_eq!(meta.identity, *c.identity());
        assert_eq!(meta.header.bytes, c.payload().len() as u64);
        let from_bytes = Capsule::inspect_bytes(&c.to_bytes()).unwrap();
        assert_eq!(from_bytes, meta);
    }
    #[test]
    fn release_reports_reclaimed_bytes() {
        let c = sample(vec![7u8; 256]);
        assert_eq!(c.release(), 256);
    }
    #[test]
    fn is_loadable_delegates_to_identity() {
        let c = sample(vec![1]);
        assert!(c.is_loadable(&identity()).is_ok());
        let mut live = identity();
        live.security_domain = "other".into();
 assert!(matches!( c.is_loadable(&live), Err(IncompatibleReason::SecurityDomain { .. }) ));
    }
    #[test]
    fn sha256_sealed_capsule_roundtrips() {
        let c = CapsuleBuilder::new(CapsuleType::Kv, "m", identity())
            .integrity_algo(IntegrityAlgo::Sha256)
            .seal(vec![3u8; 300]);
        assert_eq!(c.header().integrity.algo, IntegrityAlgo::Sha256);
        let back = Capsule::from_bytes(&c.to_bytes()).unwrap();
        assert_eq!(c, back);
    }
}
}


// --- inlined state/error.rs ---
pub mod error {
//! Typed errors for capsule schemas, serialization, compatibility, and stores.
//!
//! Every rejection is a distinct variant so callers branch on the reason
//! programmatically rather than parsing a message. In particular
//! [`IncompatibleReason`] names exactly which identity field disagreed, so a
//! loader never has to scrape a string to learn why a capsule cannot bind to a
//! live runtime.

use thiserror::Error;

/// The precise reason a sealed capsule cannot bind to a live runtime identity.
///
/// Each variant carries the capsule-side value and the live-side value for the
/// one field that disagreed, so the caller can report or reconcile without
/// re-deriving the comparison.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IncompatibleReason {
    #[error("model weights id differs: capsule {capsule:?}, live {live:?}")]
    ModelWeights { capsule: String, live: String },

    #[error("architecture id differs: capsule {capsule:?}, live {live:?}")]
    Arch { capsule: String, live: String },

    #[error("tokenizer id differs: capsule {capsule:?}, live {live:?}")]
    Tokenizer { capsule: String, live: String },

    #[error("prompt ABI version differs: capsule {capsule:?}, live {live:?}")]
    PromptAbi { capsule: String, live: String },

    #[error("tool registry id differs: capsule {capsule:?}, live {live:?}")]
    ToolRegistry { capsule: String, live: String },

    #[error("engine build id differs: capsule {capsule:?}, live {live:?}")]
    EngineBuild { capsule: String, live: String },

    #[error("security domain differs: capsule {capsule:?}, live {live:?}")]
    SecurityDomain { capsule: String, live: String },
}

/// Errors surfaced by capsule serialization, integrity, and the store impls.
#[derive(Debug, Error)]
pub enum CapsuleError {
    #[error("not a capsule byte stream: magic header did not match")]
    BadMagic,

    #[error("unsupported capsule format version {found} (this build reads {supported})")]
    UnsupportedVersion { found: u16, supported: u16 },

    #[error("capsule byte stream truncated: {detail}")]
    Truncated { detail: &'static str },

    #[error("declared payload length {declared} does not match actual {actual}")]
    LengthMismatch { declared: u64, actual: u64 },

    #[error("integrity check failed: the header digest does not match the payload")]
    IntegrityMismatch,

    #[error("content address mismatch: expected {expected:?}, computed {actual:?}")]
    ContentAddressMismatch { expected: String, actual: String },

    #[error("stored object is corrupt: {detail}")]
    Corrupt { detail: String },

    #[error("no capsule with id {0:?}")]
    NotFound(String),

    #[error("capsule metadata is not valid json: {0}")]
    Meta(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CapsuleError>;
}


// --- inlined state/header.rs ---
pub mod header {
//! Capsule kinds and the header that describes a sealed capsule.
//!
//! The header is everything a reader needs to identify a capsule and verify its
//! payload without loading the payload itself: what kind of state it holds, the
//! model and runtime it was captured under, where in the sequence it sits, its
//! ancestry, its size, and the integrity digest of its bytes.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::state::integrity::Integrity;

/// The kind of runtime state a capsule holds. Every kind shares the same header
/// and integrity story; the type only tells a consumer how to interpret the
/// opaque payload once it has been verified and bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapsuleType {
    /// Recurrent (state-space) hidden state.
    Recurrent,
    /// Attention key/value cache.
    Kv,
    /// A hybrid model carrying both a recurrent state and a key/value cache.
    HybridRecurrentKv,
    /// A reference to a shared prefix cache rather than the cache bytes.
    PrefixCacheRef,
    /// Serialized tool-runtime state.
    ToolRuntime,
    /// Serialized browser-session state.
    Browser,
    /// A repository checkpoint.
    RepoCheckpoint,
    /// A projection of a conversation into a compact carried form.
    ConversationProjection,
}

/// A capsule identifier. Minted as a ULID string so ids sort by creation time
/// and are unique without coordination.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CapsuleId(pub String);

impl CapsuleId {
    /// Mint a fresh, unique id.
    pub fn new() -> Self {
        CapsuleId(Ulid::new().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for CapsuleId {
    fn default() -> Self {
        CapsuleId::new()
    }
}

impl std::fmt::Display for CapsuleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Wall-clock milliseconds since the Unix epoch, captured once when a header is
/// sealed. Informational only; ordering between capsules uses ancestry, not
/// this timestamp.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// The self-describing header of a capsule.
///
/// `bytes` is the payload length and `integrity` is the digest of the payload,
/// so a reader can check both without trusting the byte stream that carried
/// them. `dtype` and `device` are free-form runtime tags (for example `"f16"`,
/// `"metal"`); the crate never interprets them, it only carries them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapsuleHeader {
    pub capsule_id: CapsuleId,
    pub capsule_type: CapsuleType,
    pub model_id: String,
    pub model_hash: String,
    pub runtime_version: String,
    pub dtype: String,
    pub device: String,
    /// Sequence position the capsule was captured at.
    pub position: u64,
    /// Hash of the context pack the capsule was produced against.
    pub context_pack_hash: String,
    /// The capsule this one was forked from, if any.
    pub parent_capsule_id: Option<CapsuleId>,
    /// Wall-clock milliseconds since the Unix epoch at seal time.
    pub created_at: u64,
    /// Length of the payload in bytes.
    pub bytes: u64,
    /// Digest of the payload.
    pub integrity: Integrity,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capsule_ids_are_distinct() {
        let a = CapsuleId::new();
        let b = CapsuleId::new();
        assert_ne!(a, b);
        assert_eq!(a.as_str().len(), 26);
    }
    #[test]
    fn capsule_type_roundtrips_through_json() {
        for ty in [
            CapsuleType::Recurrent,
            CapsuleType::Kv,
            CapsuleType::HybridRecurrentKv,
            CapsuleType::PrefixCacheRef,
            CapsuleType::ToolRuntime,
            CapsuleType::Browser,
            CapsuleType::RepoCheckpoint,
            CapsuleType::ConversationProjection,
        ] {
            let json = serde_json::to_string(&ty).unwrap();
            let back: CapsuleType = serde_json::from_str(&json).unwrap();
            assert_eq!(ty, back);
        }
    }
}
}


// --- inlined state/identity.rs ---
pub mod identity {
//! Identity binding: the contract a capsule must satisfy to load into a runtime.
//!
//! A capsule is only meaningful under the exact conditions it was captured
//! under. Loading recurrent or cache bytes produced under one tokenizer, one
//! prompt ABI, or one security domain into a runtime configured differently is
//! not degraded, it is undefined. So a capsule carries an [`IdentityBinding`]
//! and refuses to bind unless every field matches the live runtime's binding.
//!
//! The comparison is strict equality on every field. A real runtime may later
//! choose to relax individual fields to compatibility ranges (for example a
//! prompt-ABI minimum rather than an exact match); that policy is
//! DEFERRED_MODEL_REQUIRED and lives with the runtime, not with these schemas.

use serde::{Deserialize, Serialize};

use crate::state::error::IncompatibleReason;

/// The set of identifiers a capsule must share with a live runtime to bind.
///
/// All fields are opaque identifiers compared by equality. The crate does not
/// interpret their contents; it only checks that the sealed side and the live
/// side agree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityBinding {
    pub model_weights_id: String,
    pub arch_id: String,
    pub tokenizer_id: String,
    pub prompt_abi_version: String,
    pub tool_registry_id: String,
    pub engine_build_id: String,
    pub security_domain: String,
}

impl IdentityBinding {
    /// Check whether a capsule carrying this binding can load into a runtime
    /// whose live binding is `live`.
    ///
    /// Returns `Ok(())` when every field agrees. Otherwise returns the typed
    /// reason for the first field that disagrees. The order is fixed and puts
    /// the security domain first, so a security-domain mismatch is reported in
    /// preference to any other difference.
    pub fn is_loadable(&self, live: &IdentityBinding) -> Result<(), IncompatibleReason> {
        if self.security_domain != live.security_domain {
            return Err(IncompatibleReason::SecurityDomain {
                capsule: self.security_domain.clone(),
                live: live.security_domain.clone(),
            });
        }
        if self.model_weights_id != live.model_weights_id {
            return Err(IncompatibleReason::ModelWeights {
                capsule: self.model_weights_id.clone(),
                live: live.model_weights_id.clone(),
            });
        }
        if self.arch_id != live.arch_id {
            return Err(IncompatibleReason::Arch {
                capsule: self.arch_id.clone(),
                live: live.arch_id.clone(),
            });
        }
        if self.tokenizer_id != live.tokenizer_id {
            return Err(IncompatibleReason::Tokenizer {
                capsule: self.tokenizer_id.clone(),
                live: live.tokenizer_id.clone(),
            });
        }
        if self.prompt_abi_version != live.prompt_abi_version {
            return Err(IncompatibleReason::PromptAbi {
                capsule: self.prompt_abi_version.clone(),
                live: live.prompt_abi_version.clone(),
            });
        }
        if self.tool_registry_id != live.tool_registry_id {
            return Err(IncompatibleReason::ToolRegistry {
                capsule: self.tool_registry_id.clone(),
                live: live.tool_registry_id.clone(),
            });
        }
        if self.engine_build_id != live.engine_build_id {
            return Err(IncompatibleReason::EngineBuild {
                capsule: self.engine_build_id.clone(),
                live: live.engine_build_id.clone(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> IdentityBinding {
        IdentityBinding {
            model_weights_id: "weights-a".into(),
            arch_id: "arch-a".into(),
            tokenizer_id: "tok-a".into(),
            prompt_abi_version: "abi-1".into(),
            tool_registry_id: "reg-a".into(),
            engine_build_id: "build-a".into(),
            security_domain: "domain-a".into(),
        }
    }
    #[test]
    fn identical_bindings_are_loadable() {
        let a = sample();
        assert_eq!(a.is_loadable(&sample()), Ok(()));
    }
    #[test]
    fn each_field_mismatch_has_its_own_reason() {
        let base = sample();
        let mut live = sample();
        live.model_weights_id = "weights-b".into();
        assert_eq!(base.is_loadable(&live), Err(IncompatibleReason::ModelWeights { capsule: "weights-a".into(), live: "weights-b".into(), }));
        let mut live = sample();
        live.tokenizer_id = "tok-b".into();
        assert_eq!(base.is_loadable(&live), Err(IncompatibleReason::Tokenizer { capsule: "tok-a".into(), live: "tok-b".into(), }));
        let mut live = sample();
        live.security_domain = "domain-b".into();
        assert_eq!(base.is_loadable(&live), Err(IncompatibleReason::SecurityDomain { capsule: "domain-a".into(), live: "domain-b".into(), }));
        let mut live = sample();
        live.arch_id = "arch-b".into();
 assert!(matches!( base.is_loadable(&live), Err(IncompatibleReason::Arch { .. }) ));
        let mut live = sample();
        live.prompt_abi_version = "abi-2".into();
 assert!(matches!( base.is_loadable(&live), Err(IncompatibleReason::PromptAbi { .. }) ));
        let mut live = sample();
        live.tool_registry_id = "reg-b".into();
 assert!(matches!( base.is_loadable(&live), Err(IncompatibleReason::ToolRegistry { .. }) ));
        let mut live = sample();
        live.engine_build_id = "build-b".into();
 assert!(matches!( base.is_loadable(&live), Err(IncompatibleReason::EngineBuild { .. }) ));
    }
    #[test]
    fn security_domain_is_reported_before_other_mismatches() {
        let base = sample();
        let mut live = sample();
        live.security_domain = "domain-b".into();
        live.tokenizer_id = "tok-b".into();
 assert!(matches!( base.is_loadable(&live), Err(IncompatibleReason::SecurityDomain { .. }) ));
    }
}
}


// --- inlined state/integrity.rs ---
pub mod integrity {
//! Integrity digests for capsule payloads.
//!
//! A capsule records the digest of its payload so a reader can prove the bytes
//! it loaded are the bytes that were sealed. Both algorithms the Bible names
//! (sec 23) are supported: sha256 and blake3. Each produces a fixed 32-byte
//! digest, so the in-memory form is a `[u8; 32]` tagged by the algorithm that
//! produced it. On the wire the digest serializes as a self-describing tagged
//! hex string of the form `algo:hex`, so a serialized capsule carries the
//! algorithm alongside the bytes and never needs an out-of-band convention.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as _, Sha256};

/// The digest algorithm that produced an [`Integrity`] value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegrityAlgo {
    Sha256,
    Blake3,
}

impl IntegrityAlgo {
    /// The lowercase tag used in the serialized `algo:hex` form.
    pub fn tag(self) -> &'static str {
        match self {
            IntegrityAlgo::Sha256 => "sha256",
            IntegrityAlgo::Blake3 => "blake3",
        }
    }

    /// Parse an algorithm tag, returning `None` for an unknown tag.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "sha256" => Some(IntegrityAlgo::Sha256),
            "blake3" => Some(IntegrityAlgo::Blake3),
            _ => None,
        }
    }
}

/// A payload digest tagged by the algorithm that produced it. Both supported
/// algorithms yield 32 bytes, so the digest is a fixed array rather than a
/// variable buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Integrity {
    pub algo: IntegrityAlgo,
    pub digest: [u8; 32],
}

impl Integrity {
    /// Compute the digest of `bytes` with the given algorithm.
    pub fn compute(algo: IntegrityAlgo, bytes: &[u8]) -> Self {
        match algo {
            IntegrityAlgo::Sha256 => Self::sha256(bytes),
            IntegrityAlgo::Blake3 => Self::blake3(bytes),
        }
    }

    /// Compute a sha256 digest of `bytes`.
    pub fn sha256(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let out = hasher.finalize();
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&out);
        Integrity {
            algo: IntegrityAlgo::Sha256,
            digest,
        }
    }

    /// Compute a blake3 digest of `bytes`.
    pub fn blake3(bytes: &[u8]) -> Self {
        Integrity {
            algo: IntegrityAlgo::Blake3,
            digest: *blake3::hash(bytes).as_bytes(),
        }
    }

    /// Recompute the digest of `bytes` with this value's algorithm and return
    /// whether it matches. This is the check a reader runs to accept or reject
    /// a payload.
    pub fn verify(&self, bytes: &[u8]) -> bool {
        Self::compute(self.algo, bytes) == *self
    }

    /// Render as the self-describing `algo:hex` form.
    pub fn to_tagged_hex(&self) -> String {
        let mut s = String::with_capacity(7 + 64);
        s.push_str(self.algo.tag());
        s.push(':');
        for byte in self.digest {
            s.push_str(&format!("{byte:02x}"));
        }
        s
    }

    /// Parse the `algo:hex` form, returning `None` on any malformed input.
    pub fn from_tagged_hex(s: &str) -> Option<Self> {
        let (tag, hex) = s.split_once(':')?;
        let algo = IntegrityAlgo::from_tag(tag)?;
        if hex.len() != 64 {
            return None;
        }
        let mut digest = [0u8; 32];
        for (i, slot) in digest.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Integrity { algo, digest })
    }
}

impl Serialize for Integrity {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.to_tagged_hex())
    }
}

impl<'de> Deserialize<'de> for Integrity {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Integrity::from_tagged_hex(&s)
            .ok_or_else(|| D::Error::custom("invalid integrity tagged-hex digest"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sha256_and_blake3_differ_and_are_stable() {
        let bytes = b"synthetic capsule payload";
        let a = Integrity::sha256(bytes);
        let b = Integrity::blake3(bytes);
        assert_eq!(a.algo, IntegrityAlgo::Sha256);
        assert_eq!(b.algo, IntegrityAlgo::Blake3);
        assert_ne!(a.digest, b.digest);
        assert_eq!(a, Integrity::sha256(bytes));
        assert_eq!(b, Integrity::blake3(bytes));
    }
    #[test]
    fn verify_accepts_original_rejects_mutated() {
        let bytes = vec![1u8, 2, 3, 4, 5];
        for algo in [IntegrityAlgo::Sha256, IntegrityAlgo::Blake3] {
            let integ = Integrity::compute(algo, &bytes);
            assert!(integ.verify(&bytes));
            let mut flipped = bytes.clone();
            flipped[2] ^= 0x01;
            assert!(!integ.verify(&flipped));
        }
    }
    #[test]
    fn tagged_hex_roundtrips() {
        for algo in [IntegrityAlgo::Sha256, IntegrityAlgo::Blake3] {
            let integ = Integrity::compute(algo, b"abc");
            let text = integ.to_tagged_hex();
            assert!(text.starts_with(algo.tag()));
            assert_eq!(Integrity::from_tagged_hex(&text), Some(integ));
        }
        assert_eq!(Integrity::from_tagged_hex("nope"), None);
        assert_eq!(Integrity::from_tagged_hex("md5:00"), None);
    }
    #[test]
    fn serde_is_a_plain_string() {
        let integ = Integrity::blake3(b"payload");
        let json = serde_json::to_string(&integ).unwrap();
        assert!(json.starts_with("\"blake3:"));
        let back: Integrity = serde_json::from_str(&json).unwrap();
        assert_eq!(integ, back);
    }
}
}


// --- inlined state/store.rs ---
pub mod store {
//! Capsule stores: a trait plus an in-memory impl and a content-addressed
//! on-disk impl.
//!
//! Both impls serialize through [`Capsule::to_bytes`] and parse back through
//! [`Capsule::from_bytes`], so every load is integrity-checked. The on-disk
//! store is content-addressed: an object is named by the digest of its bytes,
//! written atomically (temp file plus rename), and a small reference file maps
//! a capsule id to its content address. A load recomputes the content address
//! and rejects a mismatch before it even checks the payload digest.

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use ulid::Ulid;

use crate::state::capsule::{Capsule, CapsuleInspect};
use crate::state::error::{CapsuleError, Result};
use crate::state::header::CapsuleId;

/// How two capsules relate through their recorded ancestry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ancestry {
    /// Both refer to the same capsule id.
    Same,
    /// The first is the parent of the second.
    ParentToChild,
    /// The second is the parent of the first.
    ChildToParent,
    /// Both were forked from the same parent.
    Siblings,
    /// No recorded relationship.
    Unrelated,
}

/// The result of comparing two capsules. Deliberately structural: it reports
/// what is and is not equal and how the two relate by ancestry, and makes no
/// judgement about which is preferable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleComparison {
    pub same_capsule_id: bool,
    pub payload_identical: bool,
    pub identity_identical: bool,
    pub header_identical: bool,
    pub ancestry: Ancestry,
}

impl CapsuleComparison {
    /// Compare two capsules field by field.
    pub fn of(a: &Capsule, b: &Capsule) -> CapsuleComparison {
        let a_id = a.capsule_id();
        let b_id = b.capsule_id();
        let a_parent = a.parent_capsule_id();
        let b_parent = b.parent_capsule_id();

        let ancestry = if a_id == b_id {
            Ancestry::Same
        } else if b_parent == Some(a_id) {
            Ancestry::ParentToChild
        } else if a_parent == Some(b_id) {
            Ancestry::ChildToParent
        } else if a_parent.is_some() && a_parent == b_parent {
            Ancestry::Siblings
        } else {
            Ancestry::Unrelated
        };

        CapsuleComparison {
            same_capsule_id: a_id == b_id,
            payload_identical: a.payload() == b.payload(),
            identity_identical: a.identity() == b.identity(),
            header_identical: a.header() == b.header(),
            ancestry,
        }
    }
}

/// A place to save, load, fork, compare, release, and inspect capsules.
///
/// `fork` and `compare` have default implementations in terms of `load` and
/// `save`, so an impl only has to provide the four primitive operations.
pub trait CapsuleStore {
    /// Save a capsule, keyed by its own id, and return that id. Saving a
    /// capsule whose id already exists overwrites the stored bytes.
    fn save(&mut self, capsule: &Capsule) -> Result<CapsuleId>;

    /// Load and integrity-check the capsule with `id`.
    fn load(&self, id: &CapsuleId) -> Result<Capsule>;

    /// Release the capsule with `id`. Returns `NotFound` if it is absent.
    fn release(&mut self, id: &CapsuleId) -> Result<()>;

    /// Inspect the metadata of the capsule with `id` without materializing its
    /// payload.
    fn inspect(&self, id: &CapsuleId) -> Result<CapsuleInspect>;

    /// Fork the capsule with `id` and save the fork, returning the new id.
    fn fork(&mut self, id: &CapsuleId) -> Result<CapsuleId> {
        let source = self.load(id)?;
        let forked = source.fork();
        self.save(&forked)
    }

    /// Compare the two stored capsules with the given ids.
    fn compare(&self, a: &CapsuleId, b: &CapsuleId) -> Result<CapsuleComparison> {
        let ca = self.load(a)?;
        let cb = self.load(b)?;
        Ok(CapsuleComparison::of(&ca, &cb))
    }
}

/// An in-memory store. Holds the serialized bytes of each capsule, so a load
/// runs the same integrity checks a persistent store would.
#[derive(Debug, Default)]
pub struct MemoryStore {
    objects: HashMap<String, Vec<u8>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        MemoryStore {
            objects: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl CapsuleStore for MemoryStore {
    fn save(&mut self, capsule: &Capsule) -> Result<CapsuleId> {
        let id = capsule.capsule_id().clone();
        self.objects.insert(id.0.clone(), capsule.to_bytes());
        Ok(id)
    }

    fn load(&self, id: &CapsuleId) -> Result<Capsule> {
        let bytes = self
            .objects
            .get(&id.0)
            .ok_or_else(|| CapsuleError::NotFound(id.0.clone()))?;
        Capsule::from_bytes(bytes)
    }

    fn release(&mut self, id: &CapsuleId) -> Result<()> {
        self.objects
            .remove(&id.0)
            .map(|_| ())
            .ok_or_else(|| CapsuleError::NotFound(id.0.clone()))
    }

    fn inspect(&self, id: &CapsuleId) -> Result<CapsuleInspect> {
        let bytes = self
            .objects
            .get(&id.0)
            .ok_or_else(|| CapsuleError::NotFound(id.0.clone()))?;
        Capsule::inspect_bytes(bytes)
    }
}

/// A content-addressed on-disk store.
///
/// Layout under `root`:
///
/// - `objects/<content-address>.capsule` holds the serialized bytes, named by
///   the blake3 digest of those bytes.
/// - `refs/<capsule-id>` holds the content address the id currently points at.
///
/// Both files are written atomically by writing a uniquely named temp file in
/// the same directory and renaming it into place. Distinct capsules with
/// identical bytes share one object; releasing an id removes its ref and, if no
/// other ref points at that object, the object too.
#[derive(Debug, Clone)]
pub struct DiskStore {
    root: PathBuf,
}

impl DiskStore {
    /// Open (creating if needed) a store rooted at `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("objects"))?;
        fs::create_dir_all(root.join("refs"))?;
        Ok(DiskStore { root })
    }

    fn objects_dir(&self) -> PathBuf {
        self.root.join("objects")
    }

    fn refs_dir(&self) -> PathBuf {
        self.root.join("refs")
    }

    fn object_path(&self, address: &str) -> PathBuf {
        self.objects_dir().join(format!("{address}.capsule"))
    }

    fn ref_path(&self, id: &CapsuleId) -> PathBuf {
        self.refs_dir().join(&id.0)
    }

    fn content_address(bytes: &[u8]) -> String {
        blake3::hash(bytes).to_hex().to_string()
    }

    /// Read a capsule id's content address from its ref file.
    fn read_ref(&self, id: &CapsuleId) -> Result<String> {
        match fs::read_to_string(self.ref_path(id)) {
            Ok(s) => Ok(s.trim().to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(CapsuleError::NotFound(id.0.clone()))
            }
            Err(e) => Err(CapsuleError::Io(e)),
        }
    }

    /// Read and content-verify the object at `address`.
    fn read_object(&self, address: &str) -> Result<Vec<u8>> {
        let path = self.object_path(address);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(CapsuleError::Corrupt {
                    detail: format!("object {address} referenced but missing"),
                });
            }
            Err(e) => return Err(CapsuleError::Io(e)),
        };
        let actual = Self::content_address(&bytes);
        if actual != address {
            return Err(CapsuleError::ContentAddressMismatch {
                expected: address.to_string(),
                actual,
            });
        }
        Ok(bytes)
    }
}

/// Write `bytes` to `path` atomically: write a uniquely named temp file in the
/// same directory, flush it, then rename it over `path`.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| CapsuleError::Corrupt {
            detail: "target path has no parent directory".to_string(),
        })?;
    let tmp = dir.join(format!(".tmp-{}", Ulid::new()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.flush()?;
    }
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Best-effort cleanup of the temp file; report the rename error.
            let _ = fs::remove_file(&tmp);
            Err(CapsuleError::Io(e))
        }
    }
}

impl CapsuleStore for DiskStore {
    fn save(&mut self, capsule: &Capsule) -> Result<CapsuleId> {
        let id = capsule.capsule_id().clone();
        let bytes = capsule.to_bytes();
        let address = Self::content_address(&bytes);
        atomic_write(&self.object_path(&address), &bytes)?;
        atomic_write(&self.ref_path(&id), address.as_bytes())?;
        Ok(id)
    }

    fn load(&self, id: &CapsuleId) -> Result<Capsule> {
        let address = self.read_ref(id)?;
        let bytes = self.read_object(&address)?;
        Capsule::from_bytes(&bytes)
    }

    fn release(&mut self, id: &CapsuleId) -> Result<()> {
        let address = self.read_ref(id)?;
        fs::remove_file(self.ref_path(id))?;
        // Garbage-collect the object if no other ref points at it.
        if !self.address_is_referenced(&address)? {
            let obj = self.object_path(&address);
            if obj.exists() {
                fs::remove_file(obj)?;
            }
        }
        Ok(())
    }

    fn inspect(&self, id: &CapsuleId) -> Result<CapsuleInspect> {
        let address = self.read_ref(id)?;
        let bytes = self.read_object(&address)?;
        Capsule::inspect_bytes(&bytes)
    }
}

impl DiskStore {
    /// Whether any ref file currently points at `address`.
    fn address_is_referenced(&self, address: &str) -> Result<bool> {
        for entry in fs::read_dir(self.refs_dir())? {
            let entry = entry?;
            if let Ok(contents) = fs::read_to_string(entry.path()) {
                if contents.trim() == address {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}
}

