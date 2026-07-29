//! # hide-objects — YOU surface object + attachment store
//!
//! Content-addressed object system for HIDE's **YOU | CHAT | IDE** surfaces.
//! All three surfaces share one session, Context OS, memory graph, object store,
//! tools, permissions and canonical events; this crate is the object store.
//!
//! ## Laws
//!
//! 1. **Content hash is identity.** Same bytes → one object, many refs.
//! 2. **Incremental processing.** Stages are independently resumable; working
//!    buffer is bounded by [`hash::CHUNK_SIZE`] (256 KiB).
//! 3. **Model sees derivatives only.** [`derivatives::CompileObjectView`] has
//!    no raw-bytes path; raw access requires [`derivatives::RawBytesCap`].
//! 4. **Queue never silent-drops.** Failures retry or land in a dead-letter
//!    log as [`queue::JobStatus::FailedVisible`].
//! 5. **Retention + permissions at read time.**
//! 6. **Storage is finite.** See [`budget::BOUND_STATEMENT`].
//!
//! ## What is fake
//!
//! OCR, ASR, and thumbnail codecs are intentionally [`processors::FakeOcrEngine`],
//! [`processors::FakeAsrEngine`], and [`processors::FakeThumbnailer`]. The
//! pipeline is real; the engines are labelled fake so nothing pretends to be a
//! model. No embeddings, no large index builds, no Metal.
//!
//! ```
//! use crate::objects::{
//!     ObjectStore, StorageBudget, ObjectSource, ObjectPermissions, Surface,
//!     RetentionPolicy, Priority, Reader, DerivativeSelection,
//! };
//! use tempfile::tempdir;
//!
//! let dir = tempdir().unwrap();
//! let store = ObjectStore::open(dir.path(), StorageBudget::test_small()).unwrap();
//! let job = store.enqueue_bytes(
//!     b"hello from YOU",
//!     "text/plain",
//!     ObjectSource::Synthetic { label: "demo".into() },
//!     ObjectPermissions::owner_only("alice", vec![Surface::You]),
//!     RetentionPolicy::durable(),
//!     Some("note.txt".into()),
//!     "alice",
//!     Priority::NORMAL,
//! ).unwrap();
//! let (_id, status) = store.process_one().unwrap();
//! assert_eq!(status, crate::objects::JobStatus::Succeeded);
//! let hash = store.hash_for_job(&job).unwrap();
//! let reader = Reader { principal: "alice".into(), surface: Surface::You };
//! let view = store.compile_view(&hash, &reader, &DerivativeSelection::default(), None).unwrap();
//! assert!(!crate::objects::CompileObjectView::exposes_raw_bytes());
//! assert!(view.try_raw_bytes().is_err());
//! let _ = view;
//! ```

pub use budget::{StorageBudget, BOUND_STATEMENT};
pub use derivatives::{
    CompileObjectView, DerivativeSelection, ModelFacingDerivative, RawBytesCap,
};
pub use error::{ObjectError, Result};
pub use hash::{ContentHash, CHUNK_SIZE};
pub use kinds::{mime_from_filename, ObjectKind};
pub use permissions::{ObjectPermissions, Reader, Surface};
pub use processors::{
    AsrEngine, FakeAsrEngine, FakeOcrEngine, FakeThumbnailer, OcrEngine, ProcessorSet,
    TextExtractor, Thumbnailer, Utf8TextExtractor,
};
pub use queue::{IngestJob, IngestQueue, JobStatus, Priority};
pub use retention::RetentionPolicy;
pub use schema::{
    Derivative, DerivativeKind, ObjectLocation, ObjectRecord, ObjectRef, ObjectSource,
    ObjectStatus, RefId, StageName, StageRecord, StageStatus,
};
pub use store::ObjectStore;

// --- inlined objects/budget.rs ---
pub mod budget {
//! Storage bounds. The system is *effectively unbounded* relative to a single
//! user turn, but never literally unlimited.
//!
//! Bounds are configuration + local/cloud capacity + model capability + user
//! policy. The schema and runtime both refuse to claim otherwise.

use serde::{Deserialize, Serialize};

use crate::objects::error::{ObjectError, Result};

/// Configured ceilings for the object store.
///
/// Defaults are intentionally modest for LIGHT_ONLY tests; production sets
/// these from user policy and measured free space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageBudget {
    /// Max total local blob bytes retained by this store instance.
    pub max_local_bytes: u64,
    /// Max total cloud-resident blob bytes (accounting only; this crate does
    /// not implement cloud I/O).
    pub max_cloud_bytes: u64,
    /// Hard cap on a single object body.
    pub max_object_bytes: u64,
    /// Human-readable policy note recorded on every rejection.
    pub policy_note: String,
}

impl Default for StorageBudget {
    fn default() -> Self {
        Self {
            // 64 GiB local default — "effectively unbounded" for personal use,
            // still a hard number the runtime enforces.
            max_local_bytes: 64 * 1024 * 1024 * 1024,
            max_cloud_bytes: 256 * 1024 * 1024 * 1024,
            // 32 GiB single-object cap (large video), still finite.
            max_object_bytes: 32 * 1024 * 1024 * 1024,
            policy_note: "bounded by configured local/cloud storage, model capability, and user policy — not unlimited".into(),
        }
    }
}

impl StorageBudget {
    /// Tight budget for unit tests.
    pub fn test_small() -> Self {
        Self {
            max_local_bytes: 64 * 1024 * 1024, // 64 MiB
            max_cloud_bytes: 64 * 1024 * 1024,
            max_object_bytes: 32 * 1024 * 1024, // 32 MiB
            policy_note: "test budget — deliberately small".into(),
        }
    }

    pub fn check_object_size(&self, size: u64) -> Result<()> {
        if size > self.max_object_bytes {
            return Err(ObjectError::ObjectTooLarge {
                size,
                max: self.max_object_bytes,
            });
        }
        Ok(())
    }

    pub fn check_local_admission(&self, used: u64, additional: u64) -> Result<()> {
        self.check_object_size(additional)?;
        let need = used.saturating_add(additional);
        if need > self.max_local_bytes {
            let available = self.max_local_bytes.saturating_sub(used);
            return Err(ObjectError::BudgetExceeded {
                need: additional,
                available,
                budget: format!(
                    "max_local_bytes={} ({})",
                    self.max_local_bytes, self.policy_note
                ),
            });
        }
        Ok(())
    }
}

/// Honest bound statement for contracts and docs.
pub const BOUND_STATEMENT: &str = "Storage is effectively unbounded relative to a single turn or attachment, but is always finite: bounded by configured local/cloud storage (StorageBudget), free disk, model context capability for derivatives, and user policy. Never claim literal unlimited storage.";
}


// --- inlined objects/derivatives.rs ---
pub mod derivatives {
//! Derivatives and the model-facing type boundary.
//!
//! The context-compile path may only receive [`ModelFacingDerivative`] /
//! [`CompileObjectView`]. It has **no** field and **no** method that yields
//! raw object bytes. Raw access requires an explicit [`RawBytesCap`] held only
//! by privileged host paths (export, local open), never by the compile path.

use serde::{Deserialize, Serialize};

use crate::objects::error::{ObjectError, Result};
use crate::objects::hash::ContentHash;
use crate::objects::kinds::ObjectKind;
use crate::objects::schema::{Derivative, DerivativeKind, ObjectRecord};

/// Capability token required to read raw object body bytes.
///
/// Construct only at privileged host entry points (export, download, open-in-
/// place). The context compiler must never hold this type.
#[derive(Debug, Clone, Copy)]
pub struct RawBytesCap {
    _private: (),
}

impl RawBytesCap {
    /// Mint only at privileged host paths — not the context-compile path.
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// A single derivative selected for model context.
///
/// Deliberately cannot carry raw body bytes: only derivative text or a
/// content-hash reference to a derivative blob.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFacingDerivative {
    pub kind: DerivativeKind,
    pub mime: String,
    /// Inline text when the derivative is small text (OCR, transcript, extract).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Content hash of a non-text derivative (thumbnail/proxy) — never the
    /// original object hash unless they happen to collide (they must not).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivative_hash: Option<ContentHash>,
    pub size_bytes: u64,
    pub produced_by: String,
}

impl ModelFacingDerivative {
    pub fn from_derivative(d: &Derivative) -> Self {
        Self {
            kind: d.kind,
            mime: d.mime.clone(),
            text: d.inline_text.clone(),
            derivative_hash: d.content_hash.clone(),
            size_bytes: d.size_bytes,
            produced_by: d.produced_by.clone(),
        }
    }
}

/// What the context-compile path is allowed to see for one object.
///
/// No raw bytes. No filesystem path to the body. Only selected derivatives
/// plus safe metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompileObjectView {
    pub content_hash: ContentHash,
    pub kind: ObjectKind,
    pub mime: String,
    pub size_bytes: u64,
    pub label: Option<String>,
    pub derivatives: Vec<ModelFacingDerivative>,
}

impl CompileObjectView {
    /// Build a compile view from a ready record and a selection of derivative kinds.
    ///
    /// Missing requested kinds are omitted (not an error); empty selection
    /// yields metadata-only.
    pub fn from_record(
        record: &ObjectRecord,
        select: &[DerivativeKind],
        label: Option<String>,
    ) -> Self {
        let derivatives = select
            .iter()
            .filter_map(|k| record.derivative(*k).map(ModelFacingDerivative::from_derivative))
            .collect();
        Self {
            content_hash: record.content_hash.clone(),
            kind: record.kind,
            mime: record.mime.clone(),
            size_bytes: record.size_bytes,
            label,
            derivatives,
        }
    }

    /// There is intentionally no `raw_bytes` / `body` method on this type.
    /// This helper documents the boundary for tests and the contract.
    pub fn exposes_raw_bytes() -> bool {
        false
    }

    /// Attempting to "upgrade" a compile view to raw bytes always fails.
    pub fn try_raw_bytes(&self) -> Result<Vec<u8>> {
        let _ = self;
        Err(ObjectError::RawBytesForbidden)
    }
}

/// Selection request from the context compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivativeSelection {
    pub kinds: Vec<DerivativeKind>,
}

impl Default for DerivativeSelection {
    fn default() -> Self {
        Self {
            kinds: vec![
                DerivativeKind::TextExtract,
                DerivativeKind::Ocr,
                DerivativeKind::Transcript,
                DerivativeKind::Summary,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::permissions::{ObjectPermissions, Surface};
    use crate::objects::retention::RetentionPolicy;
    use crate::objects::schema::*;
    fn sample_record() -> ObjectRecord {
        ObjectRecord {
            content_hash: ContentHash::of_bytes(b"body"),
            mime: "text/plain".into(),
            kind: ObjectKind::Document,
            size_bytes: 4,
            source: ObjectSource::Synthetic {
                label: "t".into(),
            },
            location: ObjectLocation::Pending,
            status: ObjectStatus::Ready,
            stages: vec![],
            derivatives: vec![Derivative {
                kind: DerivativeKind::TextExtract,
                content_hash: None,
                mime: "text/plain".into(),
                size_bytes: 4,
                inline_text: Some("body".into()),
                produced_by: "utf8_text_extract".into(),
                produced_at_ms: 0,
            }],
            permissions: ObjectPermissions::owner_only("u", vec![Surface::You]),
            retention: RetentionPolicy::durable(),
            created_at_ms: 0,
            updated_at_ms: 0,
        }
    }
    #[test]
    fn compile_view_has_derivatives_not_raw() {
        let rec = sample_record();
        let view = CompileObjectView::from_record(
            &rec,
            &[DerivativeKind::TextExtract],
            Some("note.txt".into()),
        );
        assert_eq!(view.derivatives.len(), 1);
        assert_eq!(view.derivatives[0].text.as_deref(), Some("body"));
        assert!(!CompileObjectView::exposes_raw_bytes());
 assert!(matches!( view.try_raw_bytes(), Err(ObjectError::RawBytesForbidden) ));
    }
}
}


// --- inlined objects/error.rs ---
pub mod error {
//! Typed errors for the YOU object store.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, ObjectError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ObjectError {
    #[error("object not found: {0}")]
    NotFound(String),

    #[error("reference not found: {0}")]
    RefNotFound(String),

    #[error("permission denied: {reason}")]
    PermissionDenied { reason: String },

    #[error("retention expired or not readable: {reason}")]
    RetentionDenied { reason: String },

    #[error("storage budget exceeded: need {need} bytes, available {available} under {budget}")]
    BudgetExceeded {
        need: u64,
        available: u64,
        budget: String,
    },

    #[error("object too large: {size} bytes exceeds max_object_bytes {max}")]
    ObjectTooLarge { size: u64, max: u64 },

    #[error("queue job failed visibly (not dropped): job={job_id} stage={stage}: {detail}")]
    StageFailed {
        job_id: String,
        stage: String,
        detail: String,
    },

    #[error("stage not ready: {stage} (status={status})")]
    StageNotReady { stage: String, status: String },

    #[error("raw bytes are not reachable from the context-compile path")]
    RawBytesForbidden,

    #[error("derivative not available: {kind} for {content_hash}")]
    DerivativeMissing {
        kind: String,
        content_hash: String,
    },

    #[error("content address mismatch: expected {expected}, actual {actual}")]
    ContentAddressMismatch { expected: String, actual: String },

    #[error("io: {0}")]
    Io(String),

    #[error("invalid argument: {0}")]
    Invalid(String),

    #[error("queue empty")]
    QueueEmpty,
}

impl From<std::io::Error> for ObjectError {
    fn from(e: std::io::Error) -> Self {
        ObjectError::Io(e.to_string())
    }
}
}


// --- inlined objects/hash.rs ---
pub mod hash {
//! Content hash is the object identity.
//!
//! Two ingestions of identical bytes produce one object (same [`ContentHash`])
//! and two independent [`crate::objects::schema::ObjectRef`] records pointing at it.
//! The reverse is also true: distinct bytes never share a hash under blake3.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Read;

/// Fixed streaming chunk size for hashing and persistence.
///
/// A multi-gigabyte object is never loaded whole: each stage reads at most
/// [`CHUNK_SIZE`] bytes of working buffer at a time.
pub const CHUNK_SIZE: usize = 256 * 1024; // 256 KiB

/// blake3 content hash — the sole identity of an object body.
///
/// Wire form: `blake3:<64-hex>`. Stable across processes and platforms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentHash(pub String);

impl ContentHash {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Hash the full slice. Prefer [`hash_reader`] for large bodies.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let hex = blake3::hash(bytes).to_hex();
        Self(format!("blake3:{hex}"))
    }

    /// Stream-hash a reader. Peak buffer is [`CHUNK_SIZE`] regardless of length.
    ///
    /// Returns `(hash, size_bytes, peak_buffer_bytes)` so callers can prove the
    /// streaming bound in tests.
    pub fn of_reader<R: Read>(mut reader: R) -> std::io::Result<(Self, u64, usize)> {
        let mut hasher = blake3::Hasher::new();
        let mut buf = vec![0u8; CHUNK_SIZE];
        let mut size: u64 = 0;
        let mut peak: usize = 0;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            peak = peak.max(n);
            hasher.update(&buf[..n]);
            size += n as u64;
        }
        let hex = hasher.finalize().to_hex();
        Ok((Self(format!("blake3:{hex}")), size, peak))
    }

    pub fn is_well_formed(&self) -> bool {
        self.0.starts_with("blake3:") && self.0.len() == "blake3:".len() + 64
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ContentHash {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for ContentHash {
    fn from(s: String) -> Self {
        Self(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    fn same_bytes_same_hash() {
        let a = ContentHash::of_bytes(b"hello-you-object");
        let b = ContentHash::of_bytes(b"hello-you-object");
        assert_eq!(a, b);
        assert!(a.is_well_formed());
    }
    #[test]
    fn different_bytes_different_hash() {
        let a = ContentHash::of_bytes(b"alpha");
        let b = ContentHash::of_bytes(b"beta");
        assert_ne!(a, b);
    }
    #[test]
    fn reader_matches_bytes_and_bounds_buffer() {
        let payload = vec![0xABu8; CHUNK_SIZE * 3 + 17];
        let from_slice = ContentHash::of_bytes(&payload);
        let (from_reader, size, peak) =
            ContentHash::of_reader(Cursor::new(payload.clone())).unwrap();
        assert_eq!(from_slice, from_reader);
        assert_eq!(size, payload.len() as u64);
        assert!(peak <= CHUNK_SIZE);
    }
}
}


// --- inlined objects/kinds.rs ---
pub mod kinds {
//! Object kinds and MIME helpers.

use serde::{Deserialize, Serialize};

/// First-class object kinds for the YOU surface (and shared by CHAT / IDE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind {
    Image,
    Pdf,
    Document,
    Spreadsheet,
    Slides,
    Audio,
    Video,
    Archive,
    Code,
    WebCapture,
    Asset3d,
    DesignFile,
    EmailAttachment,
    ConnectorObject,
    /// Fallback when MIME is unknown or unmapped.
    Other,
}

impl ObjectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Pdf => "pdf",
            Self::Document => "document",
            Self::Spreadsheet => "spreadsheet",
            Self::Slides => "slides",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Archive => "archive",
            Self::Code => "code",
            Self::WebCapture => "web_capture",
            Self::Asset3d => "asset_3d",
            Self::DesignFile => "design_file",
            Self::EmailAttachment => "email_attachment",
            Self::ConnectorObject => "connector_object",
            Self::Other => "other",
        }
    }

    /// All first-class kinds the schema admits.
    pub fn all_first_class() -> &'static [ObjectKind] {
        &[
            Self::Image,
            Self::Pdf,
            Self::Document,
            Self::Spreadsheet,
            Self::Slides,
            Self::Audio,
            Self::Video,
            Self::Archive,
            Self::Code,
            Self::WebCapture,
            Self::Asset3d,
            Self::DesignFile,
            Self::EmailAttachment,
            Self::ConnectorObject,
        ]
    }

    /// Infer kind from a MIME type string.
    pub fn from_mime(mime: &str) -> Self {
        let m = mime.to_ascii_lowercase();
        if m.starts_with("image/") {
            return Self::Image;
        }
        if m == "application/pdf" {
            return Self::Pdf;
        }
        if m.starts_with("audio/") {
            return Self::Audio;
        }
        if m.starts_with("video/") {
            return Self::Video;
        }
        if matches!(
            m.as_str(),
            "application/zip"
                | "application/x-tar"
                | "application/gzip"
                | "application/x-7z-compressed"
                | "application/x-rar-compressed"
        ) {
            return Self::Archive;
        }
        if m.contains("spreadsheet")
            || m.contains("excel")
            || m == "text/csv"
            || m.ends_with("sheet")
        {
            return Self::Spreadsheet;
        }
        if m.contains("presentation") || m.contains("powerpoint") {
            return Self::Slides;
        }
        if m.contains("javascript")
            || m.contains("typescript")
            || m.contains("python")
            || m.contains("rust")
            || m == "text/x-rust"
            || m == "text/x-python"
        {
            return Self::Code;
        }
        if m == "text/html" {
            return Self::WebCapture;
        }
        if m.starts_with("text/")
            || m == "application/json"
            || m == "application/xml"
            || m.contains("wordprocessing")
            || m.contains("msword")
        {
            return Self::Document;
        }
        if m.starts_with("model/") || m.contains("gltf") || m.contains("mesh") {
            return Self::Asset3d;
        }
        if m.contains("photoshop")
            || m.contains("illustrator")
            || m.contains("figma")
            || m == "application/postscript"
            || m == "image/vnd.adobe.photoshop"
        {
            return Self::DesignFile;
        }
        if m == "message/rfc822" || m.contains("email") {
            return Self::EmailAttachment;
        }
        Self::Other
    }

    /// Whether OCR or ASR is typically required before usable text is available.
    pub fn requires_ocr_or_transcript(self) -> bool {
        matches!(
            self,
            Self::Image | Self::Pdf | Self::Audio | Self::Video
        )
    }
}

/// Best-effort MIME guess from filename extension (not authoritative).
pub fn mime_from_filename(name: &str) -> String {
    let ext = name
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "md" => "text/markdown",
        "html" | "htm" => "text/html",
        "csv" => "text/csv",
        "json" => "application/json",
        "rs" => "text/x-rust",
        "py" => "text/x-python",
        "ts" => "text/typescript",
        "js" => "text/javascript",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "zip" => "application/zip",
        "tar" => "application/x-tar",
        "gz" => "application/gzip",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "glb" | "gltf" => "model/gltf-binary",
        "eml" => "message/rfc822",
        _ => "application/octet-stream",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mime_maps_to_kinds() {
        assert_eq!(ObjectKind::from_mime("image/png"), ObjectKind::Image);
        assert_eq!(ObjectKind::from_mime("application/pdf"), ObjectKind::Pdf);
        assert_eq!(ObjectKind::from_mime("video/mp4"), ObjectKind::Video);
        assert_eq!(ObjectKind::from_mime("audio/mpeg"), ObjectKind::Audio);
        assert_eq!(ObjectKind::from_mime("text/x-rust"), ObjectKind::Code);
    }
}
}


// --- inlined objects/permissions.rs ---
pub mod permissions {
//! Per-object permissions, enforced at read time.

use serde::{Deserialize, Serialize};

use crate::objects::error::{ObjectError, Result};

/// Which HIDE surface may see the object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// Private general-purpose personal AI.
    You,
    /// Coding-agent workspace.
    Chat,
    /// Visual dev environment.
    Ide,
}

impl Surface {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::You => "you",
            Self::Chat => "chat",
            Self::Ide => "ide",
        }
    }
}

/// Who is asking to read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reader {
    pub principal: String,
    pub surface: Surface,
}

/// Access control attached to every object record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectPermissions {
    /// Owner principal (always may read/write metadata).
    pub owner: String,
    /// Additional principals allowed to read.
    #[serde(default)]
    pub readers: Vec<String>,
    /// Surfaces this object is visible on. Empty = none.
    #[serde(default)]
    pub surfaces: Vec<Surface>,
    /// Whether selected derivatives may be compiled into model context.
    #[serde(default = "default_true")]
    pub allow_model_derivatives: bool,
    /// Whether raw/export paths are allowed (still requires RawBytesCap).
    #[serde(default)]
    pub allow_export: bool,
}

fn default_true() -> bool {
    true
}

impl ObjectPermissions {
    pub fn owner_only(owner: impl Into<String>, surfaces: Vec<Surface>) -> Self {
        Self {
            owner: owner.into(),
            readers: Vec::new(),
            surfaces,
            allow_model_derivatives: true,
            allow_export: false,
        }
    }

    pub fn allows_principal(&self, principal: &str) -> bool {
        principal == self.owner || self.readers.iter().any(|r| r == principal)
    }

    pub fn allows_surface(&self, surface: Surface) -> bool {
        self.surfaces.contains(&surface)
    }

    /// Read-time gate: principal + surface must both pass.
    pub fn check_read(&self, reader: &Reader) -> Result<()> {
        if !self.allows_principal(&reader.principal) {
            return Err(ObjectError::PermissionDenied {
                reason: format!(
                    "principal '{}' is not owner or listed reader",
                    reader.principal
                ),
            });
        }
        if !self.allows_surface(reader.surface) {
            return Err(ObjectError::PermissionDenied {
                reason: format!(
                    "surface '{}' is not permitted on this object",
                    reader.surface.as_str()
                ),
            });
        }
        Ok(())
    }

    pub fn check_model_derivatives(&self, reader: &Reader) -> Result<()> {
        self.check_read(reader)?;
        if !self.allow_model_derivatives {
            return Err(ObjectError::PermissionDenied {
                reason: "allow_model_derivatives is false".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owner_reads_on_allowed_surface() {
        let p = ObjectPermissions::owner_only("alice", vec![Surface::You]);
        let r = Reader {
            principal: "alice".into(),
            surface: Surface::You,
        };
        assert!(p.check_read(&r).is_ok());
    }
    #[test]
    fn stranger_denied() {
        let p = ObjectPermissions::owner_only("alice", vec![Surface::You]);
        let r = Reader {
            principal: "bob".into(),
            surface: Surface::You,
        };
 assert!(matches!( p.check_read(&r), Err(ObjectError::PermissionDenied { .. }) ));
    }
    #[test]
    fn wrong_surface_denied() {
        let p = ObjectPermissions::owner_only("alice", vec![Surface::You]);
        let r = Reader {
            principal: "alice".into(),
            surface: Surface::Chat,
        };
 assert!(matches!( p.check_read(&r), Err(ObjectError::PermissionDenied { .. }) ));
    }
}
}


// --- inlined objects/pipeline.rs ---
pub mod pipeline {
//! Incremental processing pipeline.
//!
//! Each stage is independently resumable and recorded on the [`ObjectRecord`].
//! Streaming stages never allocate more than [`crate::objects::hash::CHUNK_SIZE`] for
//! the body working buffer.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::objects::error::{ObjectError, Result};
use crate::objects::hash::{ContentHash, CHUNK_SIZE};
use crate::objects::kinds::ObjectKind;
use crate::objects::processors::ProcessorSet;
use crate::objects::schema::{
    Derivative, DerivativeKind, ObjectLocation, ObjectRecord, ObjectStatus, StageName,
    StageRecord, StageStatus,
};

/// Outcome of running stages for one object.
#[derive(Debug, Clone)]
pub struct StageOutcome {
    pub peak_buffer_bytes: usize,
    pub completed_stage: StageName,
    pub object_complete: bool,
}

fn ensure_stage(record: &mut ObjectRecord, stage: StageName, now: u64) {
    if record.stage(stage).is_none() {
        record.stages.push(StageRecord::pending(stage, now));
    }
}

fn mark_running(record: &mut ObjectRecord, stage: StageName, now: u64) {
    ensure_stage(record, stage, now);
    if let Some(st) = record.stage_mut(stage) {
        st.status = StageStatus::Running;
        st.attempts += 1;
        st.updated_at_ms = now;
    }
}

fn mark_complete(
    record: &mut ObjectRecord,
    stage: StageName,
    now: u64,
    bytes_processed: u64,
    bytes_total: Option<u64>,
    peak: usize,
) {
    if let Some(st) = record.stage_mut(stage) {
        st.bytes_processed = bytes_processed;
        st.bytes_total = bytes_total;
        st.peak_buffer_bytes = peak.max(st.peak_buffer_bytes);
        st.status = StageStatus::Complete;
        st.updated_at_ms = now;
        st.last_error = None;
    }
    record.updated_at_ms = now;
}

fn mark_skipped(record: &mut ObjectRecord, stage: StageName, now: u64, peak: usize) {
    if let Some(st) = record.stage_mut(stage) {
        st.peak_buffer_bytes = peak.max(st.peak_buffer_bytes);
        st.status = StageStatus::Skipped;
        st.updated_at_ms = now;
    }
    record.updated_at_ms = now;
}

fn mark_failed(record: &mut ObjectRecord, stage: StageName, now: u64, err: impl Into<String>) {
    if let Some(st) = record.stage_mut(stage) {
        st.status = StageStatus::Failed;
        st.last_error = Some(err.into());
        st.updated_at_ms = now;
    }
    record.updated_at_ms = now;
}

/// Stream-copy `src` → `dst`, hashing, with optional resume from `bytes_done`.
pub fn stream_persist(
    src: &Path,
    dst: &Path,
    bytes_done: u64,
) -> Result<(Option<ContentHash>, u64, usize)> {
    let mut input = File::open(src)?;
    let src_meta = input.metadata()?;
    let total = src_meta.len();

    if bytes_done > total {
        return Err(ObjectError::Invalid(format!(
            "resume offset {bytes_done} past end {total}"
        )));
    }

    input.seek(SeekFrom::Start(0))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; CHUNK_SIZE];
    let mut peak = 0usize;
    let mut read_total = 0u64;

    let mut output = if bytes_done == 0 {
        File::create(dst)?
    } else {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(dst)?;
        f.set_len(bytes_done)?;
        f.seek(SeekFrom::Start(bytes_done))?;
        f
    };

    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        peak = peak.max(n);
        hasher.update(&buf[..n]);
        let chunk_start = read_total;
        let chunk_end = read_total + n as u64;
        if chunk_end > bytes_done {
            let skip = bytes_done.saturating_sub(chunk_start) as usize;
            output.write_all(&buf[skip..n])?;
        }
        read_total = chunk_end;
    }
    output.flush()?;

    let hex = hasher.finalize().to_hex();
    let hash = ContentHash(format!("blake3:{hex}"));
    Ok((Some(hash), read_total, peak))
}

/// Hash a file without loading it whole.
pub fn hash_file(path: &Path) -> Result<(ContentHash, u64, usize)> {
    let f = File::open(path)?;
    let (h, size, peak) = ContentHash::of_reader(f)?;
    Ok((h, size, peak))
}

/// Run the next pending stage for `record`, using body at `body_path`.
pub fn run_next_stage(
    record: &mut ObjectRecord,
    body_path: &Path,
    processors: &ProcessorSet,
    now_ms: u64,
    persist_dst: Option<&Path>,
) -> Result<StageOutcome> {
    let stage = next_incomplete_stage(record).ok_or_else(|| {
        ObjectError::Invalid("no incomplete stage".into())
    })?;

    match stage {
        StageName::Receive => run_receive(record, body_path, now_ms),
        StageName::Persist => {
            let dst = persist_dst.ok_or_else(|| {
                ObjectError::Invalid("persist_dst required for Persist stage".into())
            })?;
            run_persist(record, body_path, dst, now_ms)
        }
        StageName::Classify => run_classify(record, now_ms),
        StageName::ExtractText => run_extract_text(record, body_path, processors, now_ms),
        StageName::OcrOrTranscript => run_ocr_or_transcript(record, body_path, processors, now_ms),
        StageName::Thumbnail => run_thumbnail(record, body_path, processors, now_ms),
        StageName::Finalize => run_finalize(record, now_ms),
    }
}

fn next_incomplete_stage(record: &ObjectRecord) -> Option<StageName> {
    for s in StageName::pipeline() {
        match record.stage(*s) {
            None => return Some(*s),
            Some(r)
                if matches!(
                    r.status,
                    StageStatus::Pending | StageStatus::Partial | StageStatus::Failed
                ) =>
            {
                return Some(*s);
            }
            Some(r) if matches!(r.status, StageStatus::Complete | StageStatus::Skipped) => {
                continue;
            }
            Some(_) => return Some(*s),
        }
    }
    None
}

fn run_receive(record: &mut ObjectRecord, body_path: &Path, now: u64) -> Result<StageOutcome> {
    mark_running(record, StageName::Receive, now);

    let (hash, size, peak) = hash_file(body_path)?;
    let pending = record.content_hash.as_str().is_empty()
        || record.content_hash.as_str() == "blake3:pending";
    if pending {
        record.content_hash = hash.clone();
    } else if record.content_hash != hash {
        mark_failed(record, StageName::Receive, now, "hash mismatch on receive");
        return Err(ObjectError::ContentAddressMismatch {
            expected: record.content_hash.as_str().into(),
            actual: hash.as_str().into(),
        });
    }
    record.size_bytes = size;
    record.status = ObjectStatus::Processing;
    mark_complete(record, StageName::Receive, now, size, Some(size), peak);

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::Receive,
        object_complete: false,
    })
}

fn run_persist(
    record: &mut ObjectRecord,
    body_path: &Path,
    dst: &Path,
    now: u64,
) -> Result<StageOutcome> {
    let already = record
        .stage(StageName::Persist)
        .map(|s| s.bytes_processed)
        .unwrap_or(0);
    mark_running(record, StageName::Persist, now);

    if dst.exists() {
        let (h, size, peak) = hash_file(dst)?;
        if h == record.content_hash {
            record.location = ObjectLocation::Local {
                path: dst.display().to_string(),
            };
            mark_complete(record, StageName::Persist, now, size, Some(size), peak);
            return Ok(StageOutcome {
                peak_buffer_bytes: peak,
                completed_stage: StageName::Persist,
                object_complete: false,
            });
        }
    }

    let expected = record.content_hash.clone();
    let (hash, total, peak) = stream_persist(body_path, dst, already)?;
    if let Some(h) = hash {
        if h != expected {
            mark_failed(record, StageName::Persist, now, "persist hash mismatch");
            return Err(ObjectError::ContentAddressMismatch {
                expected: expected.as_str().into(),
                actual: h.as_str().into(),
            });
        }
    }

    record.location = ObjectLocation::Local {
        path: dst.display().to_string(),
    };
    mark_complete(record, StageName::Persist, now, total, Some(total), peak);

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::Persist,
        object_complete: false,
    })
}

fn run_classify(record: &mut ObjectRecord, now: u64) -> Result<StageOutcome> {
    mark_running(record, StageName::Classify, now);
    record.kind = ObjectKind::from_mime(&record.mime);
    mark_complete(record, StageName::Classify, now, 0, None, 0);
    Ok(StageOutcome {
        peak_buffer_bytes: 0,
        completed_stage: StageName::Classify,
        object_complete: false,
    })
}

/// Read up to `max_in_memory` bytes; still scans the full file for peak tracking.
fn read_body_bounded(path: &Path, max_in_memory: usize) -> Result<(Vec<u8>, usize, u64)> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    let mut chunk = vec![0u8; CHUNK_SIZE];
    let mut peak = 0usize;
    let mut total = 0u64;
    loop {
        let n = f.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        peak = peak.max(n);
        total += n as u64;
        if buf.len() < max_in_memory {
            let take = (max_in_memory - buf.len()).min(n);
            buf.extend_from_slice(&chunk[..take]);
        }
    }
    Ok((buf, peak, total))
}

const PROCESSOR_IN_MEMORY_CAP: usize = 4 * 1024 * 1024;

fn run_extract_text(
    record: &mut ObjectRecord,
    body_path: &Path,
    processors: &ProcessorSet,
    now: u64,
) -> Result<StageOutcome> {
    mark_running(record, StageName::ExtractText, now);
    let (body, peak, total) = read_body_bounded(body_path, PROCESSOR_IN_MEMORY_CAP)?;
    let mime = record.mime.clone();

    if let Some(out) = processors.text.extract(&mime, &body) {
        record.derivatives.push(Derivative {
            kind: out.kind,
            content_hash: None,
            mime: "text/plain".into(),
            size_bytes: out.text.len() as u64,
            inline_text: Some(out.text),
            produced_by: out.produced_by,
            produced_at_ms: now,
        });
        mark_complete(record, StageName::ExtractText, now, total, Some(total), peak);
    } else {
        mark_skipped(record, StageName::ExtractText, now, peak);
    }

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::ExtractText,
        object_complete: false,
    })
}

fn run_ocr_or_transcript(
    record: &mut ObjectRecord,
    body_path: &Path,
    processors: &ProcessorSet,
    now: u64,
) -> Result<StageOutcome> {
    mark_running(record, StageName::OcrOrTranscript, now);

    if !record.kind.requires_ocr_or_transcript() {
        mark_skipped(record, StageName::OcrOrTranscript, now, 0);
        return Ok(StageOutcome {
            peak_buffer_bytes: 0,
            completed_stage: StageName::OcrOrTranscript,
            object_complete: false,
        });
    }

    let (body, peak, total) = read_body_bounded(body_path, PROCESSOR_IN_MEMORY_CAP)?;
    let mime = record.mime.clone();

    let out = processors
        .ocr
        .ocr(&mime, &body)
        .or_else(|| processors.asr.transcribe(&mime, &body));

    if let Some(out) = out {
        record.derivatives.push(Derivative {
            kind: out.kind,
            content_hash: None,
            mime: "text/plain".into(),
            size_bytes: out.text.len() as u64,
            inline_text: Some(out.text),
            produced_by: out.produced_by,
            produced_at_ms: now,
        });
        mark_complete(
            record,
            StageName::OcrOrTranscript,
            now,
            total,
            Some(total),
            peak,
        );
    } else {
        mark_skipped(record, StageName::OcrOrTranscript, now, peak);
    }

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::OcrOrTranscript,
        object_complete: false,
    })
}

fn run_thumbnail(
    record: &mut ObjectRecord,
    body_path: &Path,
    processors: &ProcessorSet,
    now: u64,
) -> Result<StageOutcome> {
    mark_running(record, StageName::Thumbnail, now);
    let (body, peak, total) = read_body_bounded(body_path, PROCESSOR_IN_MEMORY_CAP)?;
    let kind = record.kind;
    let mime = record.mime.clone();

    if let Some(out) = processors.thumb.thumbnail(kind, &mime, &body) {
        let dhash = ContentHash::of_bytes(&out.bytes);
        record.derivatives.push(Derivative {
            kind: out.kind,
            content_hash: Some(dhash),
            mime: out.mime,
            size_bytes: out.bytes.len() as u64,
            inline_text: None,
            produced_by: out.produced_by,
            produced_at_ms: now,
        });
        mark_complete(record, StageName::Thumbnail, now, total, Some(total), peak);
    } else {
        mark_skipped(record, StageName::Thumbnail, now, peak);
    }

    Ok(StageOutcome {
        peak_buffer_bytes: peak,
        completed_stage: StageName::Thumbnail,
        object_complete: false,
    })
}

fn run_finalize(record: &mut ObjectRecord, now: u64) -> Result<StageOutcome> {
    mark_running(record, StageName::Finalize, now);

    if record.derivative(DerivativeKind::Summary).is_none() {
        let summary = format!(
            "object kind={} mime={} size_bytes={} hash={}",
            record.kind.as_str(),
            record.mime,
            record.size_bytes,
            record.content_hash
        );
        record.derivatives.push(Derivative {
            kind: DerivativeKind::Summary,
            content_hash: None,
            mime: "text/plain".into(),
            size_bytes: summary.len() as u64,
            inline_text: Some(summary),
            produced_by: "finalize_summary".into(),
            produced_at_ms: now,
        });
    }

    record.status = ObjectStatus::Ready;
    mark_complete(record, StageName::Finalize, now, 0, None, 0);

    Ok(StageOutcome {
        peak_buffer_bytes: 0,
        completed_stage: StageName::Finalize,
        object_complete: true,
    })
}

/// Seed a new record with all stages Pending and a placeholder hash.
pub fn new_processing_record(
    mime: String,
    source: crate::objects::schema::ObjectSource,
    permissions: crate::objects::permissions::ObjectPermissions,
    retention: crate::objects::retention::RetentionPolicy,
    now: u64,
) -> ObjectRecord {
    let stages = StageName::pipeline()
        .iter()
        .map(|s| StageRecord::pending(*s, now))
        .collect();
    ObjectRecord {
        content_hash: ContentHash("blake3:pending".into()),
        mime,
        kind: ObjectKind::Other,
        size_bytes: 0,
        source,
        location: ObjectLocation::Pending,
        status: ObjectStatus::Queued,
        stages,
        derivatives: Vec::new(),
        permissions,
        retention,
        created_at_ms: now,
        updated_at_ms: now,
    }
}
}


// --- inlined objects/processors.rs ---
pub mod processors {
//! Processing interfaces and honestly-named fakes.
//!
//! Real OCR / ASR / vision models are out of scope. The pipeline is real; the
//! engines are labelled `Fake*` so nothing pretends to be a model.

use crate::objects::kinds::ObjectKind;
use crate::objects::schema::DerivativeKind;

/// Result of a text-oriented derivative stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextDerivativeOut {
    pub kind: DerivativeKind,
    pub text: String,
    pub produced_by: String,
}

/// Result of a binary derivative (thumbnail / proxy).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryDerivativeOut {
    pub kind: DerivativeKind,
    pub bytes: Vec<u8>,
    pub mime: String,
    pub produced_by: String,
}

/// Extract plain text when the body is already textual.
pub trait TextExtractor: Send + Sync {
    fn name(&self) -> &'static str;
    fn extract(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut>;
}

/// OCR for images / scanned PDFs. Real models are not loaded here.
pub trait OcrEngine: Send + Sync {
    fn name(&self) -> &'static str;
    fn ocr(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut>;
}

/// Speech-to-text for audio / video. Real models are not loaded here.
pub trait AsrEngine: Send + Sync {
    fn name(&self) -> &'static str;
    fn transcribe(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut>;
}

/// Thumbnail / proxy generation.
pub trait Thumbnailer: Send + Sync {
    fn name(&self) -> &'static str;
    fn thumbnail(&self, kind: ObjectKind, mime: &str, body: &[u8]) -> Option<BinaryDerivativeOut>;
}

// ---------------------------------------------------------------------------
// Real lightweight extractors (no model)
// ---------------------------------------------------------------------------

/// Deterministic UTF-8 text extract for `text/*` and a few code MIME types.
#[derive(Debug, Default, Clone, Copy)]
pub struct Utf8TextExtractor;

impl TextExtractor for Utf8TextExtractor {
    fn name(&self) -> &'static str {
        "utf8_text_extract"
    }

    fn extract(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut> {
        let m = mime.to_ascii_lowercase();
        let textual = m.starts_with("text/")
            || m == "application/json"
            || m == "application/xml"
            || m.contains("javascript")
            || m.contains("typescript")
            || m.contains("python")
            || m.contains("rust");
        if !textual {
            return None;
        }
        let text = String::from_utf8_lossy(body).into_owned();
        Some(TextDerivativeOut {
            kind: DerivativeKind::TextExtract,
            text,
            produced_by: self.name().into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Honestly-named fakes
// ---------------------------------------------------------------------------

/// Fake OCR. Labels itself `FakeOcrEngine`. Not a real vision model.
///
/// Produces a deterministic placeholder string from a blake3 prefix of the body
/// so tests can assert non-empty OCR without loading models.
#[derive(Debug, Default, Clone, Copy)]
pub struct FakeOcrEngine;

impl OcrEngine for FakeOcrEngine {
    fn name(&self) -> &'static str {
        "FakeOcrEngine"
    }

    fn ocr(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut> {
        let m = mime.to_ascii_lowercase();
        if !(m.starts_with("image/") || m == "application/pdf") {
            return None;
        }
        let digest = blake3::hash(body).to_hex();
        let text = format!(
            "[FakeOcrEngine] placeholder OCR for {mime} (blake3_prefix={})",
            &digest[..12]
        );
        Some(TextDerivativeOut {
            kind: DerivativeKind::Ocr,
            text,
            produced_by: self.name().into(),
        })
    }
}

/// Fake ASR. Labels itself `FakeAsrEngine`. Not a real speech model.
#[derive(Debug, Default, Clone, Copy)]
pub struct FakeAsrEngine;

impl AsrEngine for FakeAsrEngine {
    fn name(&self) -> &'static str {
        "FakeAsrEngine"
    }

    fn transcribe(&self, mime: &str, body: &[u8]) -> Option<TextDerivativeOut> {
        let m = mime.to_ascii_lowercase();
        if !(m.starts_with("audio/") || m.starts_with("video/")) {
            return None;
        }
        let digest = blake3::hash(body).to_hex();
        let text = format!(
            "[FakeAsrEngine] placeholder transcript for {mime} (blake3_prefix={})",
            &digest[..12]
        );
        Some(TextDerivativeOut {
            kind: DerivativeKind::Transcript,
            text,
            produced_by: self.name().into(),
        })
    }
}

/// Fake thumbnailer. Labels itself `FakeThumbnailer`. Emits a tiny deterministic
/// "proxy" blob, not a real image codec.
#[derive(Debug, Default, Clone, Copy)]
pub struct FakeThumbnailer;

impl Thumbnailer for FakeThumbnailer {
    fn name(&self) -> &'static str {
        "FakeThumbnailer"
    }

    fn thumbnail(
        &self,
        kind: ObjectKind,
        mime: &str,
        body: &[u8],
    ) -> Option<BinaryDerivativeOut> {
        // Only for kinds that usually get a visual proxy.
        if !matches!(
            kind,
            ObjectKind::Image
                | ObjectKind::Pdf
                | ObjectKind::Video
                | ObjectKind::DesignFile
                | ObjectKind::Slides
        ) {
            return None;
        }
        let digest = blake3::hash(body);
        let mut bytes = b"FAKE_THUMB:".to_vec();
        bytes.extend_from_slice(digest.as_bytes());
        bytes.extend_from_slice(mime.as_bytes());
        Some(BinaryDerivativeOut {
            kind: DerivativeKind::Thumbnail,
            bytes,
            mime: "application/x-fake-thumbnail".into(),
            produced_by: self.name().into(),
        })
    }
}

/// Bundle of processors used by the pipeline.
pub struct ProcessorSet {
    pub text: Box<dyn TextExtractor>,
    pub ocr: Box<dyn OcrEngine>,
    pub asr: Box<dyn AsrEngine>,
    pub thumb: Box<dyn Thumbnailer>,
}

impl Default for ProcessorSet {
    fn default() -> Self {
        Self {
            text: Box::new(Utf8TextExtractor),
            ocr: Box::new(FakeOcrEngine),
            asr: Box::new(FakeAsrEngine),
            thumb: Box::new(FakeThumbnailer),
        }
    }
}

impl ProcessorSet {
    pub fn fake_defaults() -> Self {
        Self::default()
    }
}
}


// --- inlined objects/queue.rs ---
pub mod queue {
//! Ingestion queue: priority, retries, visible failure. Never silent drop.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use ulid::Ulid;

use crate::objects::error::{ObjectError, Result};
use crate::objects::hash::ContentHash;
use crate::objects::permissions::ObjectPermissions;
use crate::objects::retention::RetentionPolicy;
use crate::objects::schema::{ObjectSource, StageName};

/// Higher numbers run first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Priority(pub u8);

impl Priority {
    pub const LOW: Priority = Priority(10);
    pub const NORMAL: Priority = Priority(50);
    pub const HIGH: Priority = Priority(80);
    pub const CRITICAL: Priority = Priority(100);
}

/// Status of a queued ingestion job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    /// Waiting to retry a failed stage.
    RetryWait,
    /// All stages complete.
    Succeeded,
    /// Exhausted retries — failed **visibly**, still on the dead-letter log.
    FailedVisible,
}

/// One ingestion job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestJob {
    pub id: String,
    pub priority: Priority,
    /// Sequence for FIFO within the same priority (lower = older = first).
    pub seq: u64,
    pub status: JobStatus,
    pub mime: String,
    pub source: ObjectSource,
    pub permissions: ObjectPermissions,
    pub retention: RetentionPolicy,
    pub label: Option<String>,
    pub created_by: String,
    /// Filled once Receive/hash completes.
    pub content_hash: Option<ContentHash>,
    /// Next stage to run (or retry).
    pub next_stage: StageName,
    pub attempts: u32,
    pub max_attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl IngestJob {
    pub fn new(
        priority: Priority,
        seq: u64,
        mime: String,
        source: ObjectSource,
        permissions: ObjectPermissions,
        retention: RetentionPolicy,
        label: Option<String>,
        created_by: String,
        now_ms: u64,
    ) -> Self {
        Self {
            id: format!("ijob_{}", Ulid::new()),
            priority,
            seq,
            status: JobStatus::Queued,
            mime,
            source,
            permissions,
            retention,
            label,
            created_by,
            content_hash: None,
            next_stage: StageName::Receive,
            attempts: 0,
            max_attempts: 3,
            last_error: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        }
    }
}

/// Heap entry: higher priority first; for equal priority, lower seq first.
#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapEntry {
    priority: Priority,
    seq: u64,
    job_id: String,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.seq.cmp(&self.seq),
            o => o,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority queue + dead-letter for visible failures.
#[derive(Debug, Default)]
pub struct IngestQueue {
    heap: BinaryHeap<HeapEntry>,
    jobs: std::collections::BTreeMap<String, IngestJob>,
    /// Jobs that exhausted retries — never silently dropped.
    dead_letter: Vec<IngestJob>,
    next_seq: u64,
}

impl IngestQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&mut self, mut job: IngestJob) -> String {
        if job.seq == 0 {
            job.seq = self.next_seq;
            self.next_seq += 1;
        } else {
            self.next_seq = self.next_seq.max(job.seq + 1);
        }
        let id = job.id.clone();
        self.heap.push(HeapEntry {
            priority: job.priority,
            seq: job.seq,
            job_id: id.clone(),
        });
        self.jobs.insert(id.clone(), job);
        id
    }

    pub fn len(&self) -> usize {
        self.jobs
            .values()
            .filter(|j| matches!(j.status, JobStatus::Queued | JobStatus::RetryWait))
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, id: &str) -> Option<&IngestJob> {
        self.jobs.get(id)
    }

    pub fn dead_letter(&self) -> &[IngestJob] {
        &self.dead_letter
    }

    /// Pop the highest-priority ready job. Returns error only if queue empty
    /// of runnable jobs — never drops a failed job.
    pub fn pop_ready(&mut self) -> Result<IngestJob> {
        while let Some(entry) = self.heap.pop() {
            if let Some(job) = self.jobs.get(&entry.job_id) {
                if matches!(job.status, JobStatus::Queued | JobStatus::RetryWait) {
                    let mut job = self.jobs.remove(&entry.job_id).unwrap();
                    job.status = JobStatus::Running;
                    return Ok(job);
                }
            }
        }
        Err(ObjectError::QueueEmpty)
    }

    /// Re-queue after a partial stage (resume) or for the next stage.
    pub fn requeue(&mut self, mut job: IngestJob, now_ms: u64) {
        job.status = JobStatus::Queued;
        job.updated_at_ms = now_ms;
        let id = job.id.clone();
        self.heap.push(HeapEntry {
            priority: job.priority,
            seq: job.seq,
            job_id: id.clone(),
        });
        self.jobs.insert(id, job);
    }

    /// Record a retryable stage failure. If attempts exhausted, move to dead
    /// letter with [`JobStatus::FailedVisible`] — never silent drop.
    pub fn fail_stage(
        &mut self,
        mut job: IngestJob,
        stage: StageName,
        detail: impl Into<String>,
        now_ms: u64,
    ) -> JobStatus {
        let detail = detail.into();
        job.attempts += 1;
        job.last_error = Some(format!("{}: {detail}", stage.as_str()));
        job.updated_at_ms = now_ms;
        job.next_stage = stage;

        if job.attempts >= job.max_attempts {
            job.status = JobStatus::FailedVisible;
            let status = job.status;
            self.dead_letter.push(job);
            status
        } else {
            job.status = JobStatus::RetryWait;
            let status = job.status;
            let id = job.id.clone();
            self.heap.push(HeapEntry {
                priority: job.priority,
                seq: job.seq,
                job_id: id.clone(),
            });
            self.jobs.insert(id, job);
            status
        }
    }

    pub fn complete(&mut self, mut job: IngestJob, now_ms: u64) {
        job.status = JobStatus::Succeeded;
        job.updated_at_ms = now_ms;
        // Keep terminal jobs in the map for inspection.
        self.jobs.insert(job.id.clone(), job);
    }

    pub fn put_running(&mut self, job: IngestJob) {
        self.jobs.insert(job.id.clone(), job);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::permissions::{ObjectPermissions, Surface};
    fn job(pri: Priority, seq: u64) -> IngestJob {
        IngestJob::new(
            pri,
            seq,
            "text/plain".into(),
            ObjectSource::Synthetic {
                label: "t".into(),
            },
            ObjectPermissions::owner_only("u", vec![Surface::You]),
            RetentionPolicy::durable(),
            None,
            "u".into(),
            0,
        )
    }
    #[test]
    fn higher_priority_first() {
        let mut q = IngestQueue::new();
        let low = job(Priority::LOW, 1);
        let high = job(Priority::HIGH, 2);
        let low_id = low.id.clone();
        let high_id = high.id.clone();
        q.enqueue(low);
        q.enqueue(high);
        let first = q.pop_ready().unwrap();
        assert_eq!(first.id, high_id);
        let second = q.pop_ready().unwrap();
        assert_eq!(second.id, low_id);
    }
    #[test]
    fn exhausted_retries_go_to_dead_letter_not_dropped() {
        let mut q = IngestQueue::new();
        let mut j = job(Priority::NORMAL, 1);
        j.max_attempts = 2;
        let id = j.id.clone();
        q.enqueue(j);
        let j = q.pop_ready().unwrap();
        let st = q.fail_stage(j, StageName::Persist, "disk full", 1);
        assert_eq!(st, JobStatus::RetryWait);
        let j = q.pop_ready().unwrap();
        assert_eq!(j.id, id);
        let st = q.fail_stage(j, StageName::Persist, "disk full again", 2);
        assert_eq!(st, JobStatus::FailedVisible);
        assert_eq!(q.dead_letter().len(), 1);
        assert_eq!(q.dead_letter()[0].id, id);
        assert!(q.is_empty());
    }
}
}


// --- inlined objects/retention.rs ---
pub mod retention {
//! Per-object retention, enforced at read time.

use serde::{Deserialize, Serialize};

use crate::objects::error::{ObjectError, Result};

/// How long an object remains readable.
///
/// The store never silently drops objects on TTL expiry: reads fail visibly
/// with [`ObjectError::RetentionDenied`]. Explicit GC is a separate path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum RetentionPolicy {
    /// Readable only while the named session is considered live.
    Session { session_id: String },
    /// Survives sessions; explicit delete only.
    Durable,
    /// Wall-clock expiry (ms since epoch). After this, reads fail.
    Ttl { expires_at_ms: u64 },
    /// Never auto-expire; only owner delete.
    ExplicitDeleteOnly,
}

impl RetentionPolicy {
    pub fn durable() -> Self {
        Self::Durable
    }

    pub fn session(session_id: impl Into<String>) -> Self {
        Self::Session {
            session_id: session_id.into(),
        }
    }

    pub fn ttl_until(expires_at_ms: u64) -> Self {
        Self::Ttl { expires_at_ms }
    }

    /// Read-time check.
    ///
    /// - `now_ms`: wall clock for TTL
    /// - `live_session`: if `Some`, session-scoped objects for that id are live
    pub fn check_readable(&self, now_ms: u64, live_session: Option<&str>) -> Result<()> {
        match self {
            Self::Durable | Self::ExplicitDeleteOnly => Ok(()),
            Self::Session { session_id } => match live_session {
                Some(live) if live == session_id => Ok(()),
                Some(_) => Err(ObjectError::RetentionDenied {
                    reason: format!(
                        "session-scoped object for '{session_id}' not live (active={live_session:?})"
                    ),
                }),
                None => Err(ObjectError::RetentionDenied {
                    reason: format!(
                        "session-scoped object for '{session_id}' has no live session"
                    ),
                }),
            },
            Self::Ttl { expires_at_ms } => {
                if now_ms >= *expires_at_ms {
                    Err(ObjectError::RetentionDenied {
                        reason: format!(
                            "ttl expired at {expires_at_ms} (now={now_ms})"
                        ),
                    })
                } else {
                    Ok(())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ttl_blocks_after_expiry() {
        let r = RetentionPolicy::ttl_until(1000);
        assert!(r.check_readable(999, None).is_ok());
 assert!(matches!( r.check_readable(1000, None), Err(ObjectError::RetentionDenied { .. }) ));
    }
    #[test]
    fn session_requires_live() {
        let r = RetentionPolicy::session("ses_1");
        assert!(r.check_readable(0, Some("ses_1")).is_ok());
        assert!(r.check_readable(0, Some("ses_other")).is_err());
        assert!(r.check_readable(0, None).is_err());
    }
}
}


// --- inlined objects/schema.rs ---
pub mod schema {
//! Object record schema: the durable shape of a YOU-store object.

use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::objects::hash::ContentHash;
use crate::objects::kinds::ObjectKind;
use crate::objects::permissions::ObjectPermissions;
use crate::objects::retention::RetentionPolicy;

/// Stable reference id (not the content identity). Many refs may share one hash.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RefId(pub String);

impl RefId {
    pub fn new() -> Self {
        Self(format!("oref_{}", Ulid::new()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RefId {
    fn default() -> Self {
        Self::new()
    }
}

/// Where the body lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "tier", rename_all = "snake_case")]
pub enum ObjectLocation {
    /// Content-addressed path under the local store root.
    Local { path: String },
    /// Cloud URI (accounting / future wire; not fetched by this crate).
    Cloud { uri: String },
    /// Not yet persisted.
    Pending,
}

/// Provenance of how the bytes entered the store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObjectSource {
    UserUpload {
        filename: Option<String>,
        session_id: Option<String>,
    },
    Clipboard {
        session_id: Option<String>,
    },
    ToolOutput {
        tool: String,
        call_id: Option<String>,
    },
    WebCapture {
        url: String,
    },
    Email {
        message_id: Option<String>,
        attachment_name: Option<String>,
    },
    Connector {
        connector_id: String,
        remote_id: String,
    },
    Synthetic {
        /// Test / pipeline fixtures.
        label: String,
    },
}

/// Kind of derived representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeKind {
    /// Plain text extracted without OCR/ASR.
    TextExtract,
    /// OCR text (images/PDFs).
    Ocr,
    /// Audio/video transcript.
    Transcript,
    /// Small image proxy / thumbnail.
    Thumbnail,
    /// Lower-bitrate / shorter proxy of media.
    Proxy,
    /// Short model-facing summary of the object (metadata-only here).
    Summary,
}

impl DerivativeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TextExtract => "text_extract",
            Self::Ocr => "ocr",
            Self::Transcript => "transcript",
            Self::Thumbnail => "thumbnail",
            Self::Proxy => "proxy",
            Self::Summary => "summary",
        }
    }
}

/// A derived representation. Bytes live content-addressed; this is metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Derivative {
    pub kind: DerivativeKind,
    /// Content hash of the derivative body (when materialised).
    pub content_hash: Option<ContentHash>,
    pub mime: String,
    pub size_bytes: u64,
    /// Inline text for small text derivatives (OCR/transcript/extract).
    /// Large bodies use content_hash only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline_text: Option<String>,
    /// Producer label (e.g. "FakeOcrEngine", "utf8_text_extract").
    pub produced_by: String,
    pub produced_at_ms: u64,
}

/// Pipeline stage names. Order is the default processing sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageName {
    Receive,
    Persist,
    Classify,
    ExtractText,
    OcrOrTranscript,
    Thumbnail,
    Finalize,
}

impl StageName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Receive => "receive",
            Self::Persist => "persist",
            Self::Classify => "classify",
            Self::ExtractText => "extract_text",
            Self::OcrOrTranscript => "ocr_or_transcript",
            Self::Thumbnail => "thumbnail",
            Self::Finalize => "finalize",
        }
    }

    /// Canonical ordered pipeline.
    pub fn pipeline() -> &'static [StageName] {
        &[
            Self::Receive,
            Self::Persist,
            Self::Classify,
            Self::ExtractText,
            Self::OcrOrTranscript,
            Self::Thumbnail,
            Self::Finalize,
        ]
    }
}

/// Per-stage status — independently resumable and recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Pending,
    Running,
    /// Partial progress; resume from `bytes_processed`.
    Partial,
    Complete,
    Failed,
    Skipped,
}

/// Recorded progress for one stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRecord {
    pub stage: StageName,
    pub status: StageStatus,
    /// Bytes consumed so far (for streaming stages).
    pub bytes_processed: u64,
    /// Total size if known.
    pub bytes_total: Option<u64>,
    /// Peak working buffer used during this stage (proves streaming bound).
    pub peak_buffer_bytes: usize,
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub updated_at_ms: u64,
}

impl StageRecord {
    pub fn pending(stage: StageName, now_ms: u64) -> Self {
        Self {
            stage,
            status: StageStatus::Pending,
            bytes_processed: 0,
            bytes_total: None,
            peak_buffer_bytes: 0,
            attempts: 0,
            last_error: None,
            updated_at_ms: now_ms,
        }
    }
}

/// Lifecycle of the object body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectStatus {
    Queued,
    Processing,
    Ready,
    Failed,
}

/// One content-addressed object. Identity is [`ObjectRecord::content_hash`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectRecord {
    /// blake3 of body bytes — the identity.
    pub content_hash: ContentHash,
    pub mime: String,
    pub kind: ObjectKind,
    pub size_bytes: u64,
    pub source: ObjectSource,
    pub location: ObjectLocation,
    pub status: ObjectStatus,
    pub stages: Vec<StageRecord>,
    pub derivatives: Vec<Derivative>,
    pub permissions: ObjectPermissions,
    pub retention: RetentionPolicy,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl ObjectRecord {
    pub fn stage_mut(&mut self, name: StageName) -> Option<&mut StageRecord> {
        self.stages.iter_mut().find(|s| s.stage == name)
    }

    pub fn stage(&self, name: StageName) -> Option<&StageRecord> {
        self.stages.iter().find(|s| s.stage == name)
    }

    pub fn is_ready(&self) -> bool {
        self.status == ObjectStatus::Ready
    }

    pub fn derivative(&self, kind: DerivativeKind) -> Option<&Derivative> {
        self.derivatives.iter().find(|d| d.kind == kind)
    }
}

/// A named reference to an object. Dedup is "one object, many refs".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectRef {
    pub id: RefId,
    pub content_hash: ContentHash,
    /// Display name (filename etc.) — not part of identity.
    pub label: Option<String>,
    pub created_at_ms: u64,
    /// Principal that created this reference.
    pub created_by: String,
}
}


// --- inlined objects/store.rs ---
pub mod store {
//! Content-addressed object store.
//!
//! Identity is [`ContentHash`]. The same bytes ingested twice yield one
//! [`ObjectRecord`] and two [`ObjectRef`]s.

use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::objects::budget::StorageBudget;
use crate::objects::derivatives::{CompileObjectView, DerivativeSelection, RawBytesCap};
use crate::objects::error::{ObjectError, Result};
use crate::objects::hash::{ContentHash, CHUNK_SIZE};
use crate::objects::permissions::{ObjectPermissions, Reader};
use crate::objects::pipeline::{self, new_processing_record};
use crate::objects::processors::ProcessorSet;
use crate::objects::queue::{IngestJob, IngestQueue, JobStatus, Priority};
use crate::objects::retention::RetentionPolicy;
use crate::objects::schema::{
    ObjectLocation, ObjectRecord, ObjectRef, ObjectSource, ObjectStatus, RefId, StageName,
};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

struct Inner {
    root: PathBuf,
    budget: StorageBudget,
    /// content_hash string → record
    objects: BTreeMap<String, ObjectRecord>,
    /// ref_id → ObjectRef
    refs: BTreeMap<String, ObjectRef>,
    /// job_id → staging path
    staging: BTreeMap<String, PathBuf>,
    /// job_id → in-progress ObjectRecord (before hash known / after)
    in_flight: BTreeMap<String, ObjectRecord>,
    used_local_bytes: u64,
    queue: IngestQueue,
    live_session: Option<String>,
    clock_ms: Option<u64>,
}

impl Inner {
    fn now(&self) -> u64 {
        self.clock_ms.unwrap_or_else(now_ms)
    }

    fn objects_dir(&self) -> PathBuf {
        self.root.join("objects")
    }

    fn blob_path(&self, hash: &ContentHash) -> PathBuf {
        let name = hash.as_str().trim_start_matches("blake3:");
        self.objects_dir().join(name)
    }

    fn staging_path(&self, job_id: &str) -> PathBuf {
        self.root.join("staging").join(job_id)
    }
}

/// The YOU object store: queue, content-addressed blobs, pipeline, compile view.
pub struct ObjectStore {
    inner: Arc<Mutex<Inner>>,
    processors: ProcessorSet,
}

impl ObjectStore {
    /// Open (or create) a store at `root` with the given budget.
    pub fn open(root: impl AsRef<Path>, budget: StorageBudget) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("objects"))?;
        fs::create_dir_all(root.join("derivatives"))?;
        fs::create_dir_all(root.join("staging"))?;
        fs::create_dir_all(root.join("meta"))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                root,
                budget,
                objects: BTreeMap::new(),
                refs: BTreeMap::new(),
                staging: BTreeMap::new(),
                in_flight: BTreeMap::new(),
                used_local_bytes: 0,
                queue: IngestQueue::new(),
                live_session: None,
                clock_ms: None,
            })),
            processors: ProcessorSet::fake_defaults(),
        })
    }

    pub fn set_clock_ms(&self, ms: Option<u64>) {
        self.inner.lock().clock_ms = ms;
    }

    pub fn set_live_session(&self, session_id: Option<String>) {
        self.inner.lock().live_session = session_id;
    }

    pub fn used_local_bytes(&self) -> u64 {
        self.inner.lock().used_local_bytes
    }

    pub fn budget(&self) -> StorageBudget {
        self.inner.lock().budget.clone()
    }

    pub fn queue_len(&self) -> usize {
        self.inner.lock().queue.len()
    }

    pub fn dead_letter(&self) -> Vec<IngestJob> {
        self.inner.lock().queue.dead_letter().to_vec()
    }

    // ------------------------------------------------------------------
    // Ingest API
    // ------------------------------------------------------------------

    /// Enqueue a byte body. Staging is written in [`CHUNK_SIZE`] slices so the
    /// caller can pass a large slice without the store keeping a second copy
    /// beyond the staging file.
    pub fn enqueue_bytes(
        &self,
        bytes: &[u8],
        mime: impl Into<String>,
        source: ObjectSource,
        permissions: ObjectPermissions,
        retention: RetentionPolicy,
        label: Option<String>,
        created_by: impl Into<String>,
        priority: Priority,
    ) -> Result<String> {
        let mime = mime.into();
        let created_by = created_by.into();
        let mut g = self.inner.lock();
        let now = g.now();
        g.budget.check_object_size(bytes.len() as u64)?;

        let job = IngestJob::new(
            priority,
            0,
            mime.clone(),
            source.clone(),
            permissions.clone(),
            retention.clone(),
            label,
            created_by,
            now,
        );
        let job_id = job.id.clone();
        let stage_path = g.staging_path(&job_id);
        write_bytes_chunked(&stage_path, bytes)?;
        g.staging.insert(job_id.clone(), stage_path);
        g.in_flight.insert(
            job_id.clone(),
            new_processing_record(mime, source, permissions, retention, now),
        );
        g.queue.enqueue(job);
        Ok(job_id)
    }

    /// Enqueue from an existing file path (streamed; not loaded whole into RAM
    /// by the store beyond chunked copy into staging).
    pub fn enqueue_path(
        &self,
        path: impl AsRef<Path>,
        mime: impl Into<String>,
        source: ObjectSource,
        permissions: ObjectPermissions,
        retention: RetentionPolicy,
        label: Option<String>,
        created_by: impl Into<String>,
        priority: Priority,
    ) -> Result<String> {
        let mime = mime.into();
        let created_by = created_by.into();
        let src = path.as_ref();
        let meta = fs::metadata(src)?;
        let mut g = self.inner.lock();
        let now = g.now();
        g.budget.check_object_size(meta.len())?;

        let job = IngestJob::new(
            priority,
            0,
            mime.clone(),
            source.clone(),
            permissions.clone(),
            retention.clone(),
            label,
            created_by,
            now,
        );
        let job_id = job.id.clone();
        let stage_path = g.staging_path(&job_id);
        copy_chunked(src, &stage_path)?;
        g.staging.insert(job_id.clone(), stage_path);
        g.in_flight.insert(
            job_id.clone(),
            new_processing_record(mime, source, permissions, retention, now),
        );
        g.queue.enqueue(job);
        Ok(job_id)
    }

    /// Process one job through **all** remaining stages (or until a stage fails).
    /// Returns the job id and final job status.
    pub fn process_one(&self) -> Result<(String, JobStatus)> {
        let mut job = {
            let mut g = self.inner.lock();
            g.queue.pop_ready()?
        };
        let job_id = job.id.clone();

        loop {
            let outcome = self.run_job_stage(&mut job);
            match outcome {
                Ok(done) if done => {
                    let mut g = self.inner.lock();
                    let now = g.now();
                    // Merge in-flight record into objects map (dedup by hash).
                    if let Some(rec) = g.in_flight.remove(&job_id) {
                        let hash_key = rec.content_hash.as_str().to_string();
                        job.content_hash = Some(rec.content_hash.clone());
                        let is_new = !g.objects.contains_key(&hash_key);
                        if is_new {
                            // Account storage only for new bodies.
                            if let Err(e) = g
                                .budget
                                .check_local_admission(g.used_local_bytes, rec.size_bytes)
                            {
                                // Keep in_flight so retries can re-attempt admission.
                                g.in_flight.insert(job_id.clone(), rec);
                                let st = g.queue.fail_stage(
                                    job,
                                    StageName::Finalize,
                                    e.to_string(),
                                    now,
                                );
                                return Ok((job_id, st));
                            }
                            g.used_local_bytes =
                                g.used_local_bytes.saturating_add(rec.size_bytes);
                            g.objects.insert(hash_key.clone(), rec);
                        }
                        // Always create a new ref (dedup = same object, extra ref).
                        let r = ObjectRef {
                            id: RefId::new(),
                            content_hash: ContentHash(hash_key),
                            label: job.label.clone(),
                            created_at_ms: now,
                            created_by: job.created_by.clone(),
                        };
                        g.refs.insert(r.id.as_str().to_string(), r);
                        if let Some(p) = g.staging.remove(&job_id) {
                            let _ = fs::remove_file(p);
                        }
                    }
                    g.queue.complete(job, now);
                    return Ok((job_id, JobStatus::Succeeded));
                }
                Ok(_) => {
                    // Continue to next stage in this call (full drain of one job).
                    continue;
                }
                Err(e) => {
                    let mut g = self.inner.lock();
                    let now = g.now();
                    let stage = job.next_stage;
                    let st = g.queue.fail_stage(job, stage, e.to_string(), now);
                    return Ok((job_id, st));
                }
            }
        }
    }

    /// Drain the queue until empty or a job is only RetryWait/FailedVisible.
    /// Processes Succeeded and re-attempts RetryWait jobs once per call cycle.
    pub fn drain(&self) -> Result<Vec<(String, JobStatus)>> {
        let mut out = Vec::new();
        loop {
            match self.process_one() {
                Ok(pair) => {
                    let terminal = matches!(
                        pair.1,
                        JobStatus::Succeeded | JobStatus::FailedVisible
                    );
                    out.push(pair);
                    if !terminal {
                        // RetryWait requeued — keep going if something is ready.
                        continue;
                    }
                    if self.queue_len() == 0 {
                        break;
                    }
                }
                Err(ObjectError::QueueEmpty) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }

    fn run_job_stage(&self, job: &mut IngestJob) -> Result<bool> {
        let g = self.inner.lock();
        let now = g.now();
        let stage_path = g
            .staging
            .get(&job.id)
            .cloned()
            .ok_or_else(|| ObjectError::Invalid(format!("no staging for {}", job.id)))?;
        let mut record = g
            .in_flight
            .get(&job.id)
            .cloned()
            .ok_or_else(|| ObjectError::Invalid(format!("no in_flight for {}", job.id)))?;

        // Determine next stage from record.
        let next = StageName::pipeline()
            .iter()
            .find(|s| {
                record
                    .stage(**s)
                    .map(|r| {
                        matches!(
                            r.status,
                            crate::objects::schema::StageStatus::Pending
                                | crate::objects::schema::StageStatus::Partial
                                | crate::objects::schema::StageStatus::Failed
                        )
                    })
                    .unwrap_or(true)
            })
            .copied();

        let Some(stage) = next else {
            return Ok(true);
        };
        job.next_stage = stage;

        let hash_for_persist = record.content_hash.clone();
        let persist_dst = if stage == StageName::Persist {
            if hash_for_persist.as_str() == "blake3:pending" {
                return Err(ObjectError::StageNotReady {
                    stage: "persist".into(),
                    status: "receive incomplete".into(),
                });
            }
            Some(g.blob_path(&hash_for_persist))
        } else {
            None
        };
        drop(g);

        let outcome = pipeline::run_next_stage(
            &mut record,
            &stage_path,
            &self.processors,
            now,
            persist_dst.as_deref(),
        )?;

        let mut g = self.inner.lock();
        // Write back record.
        if let Some(slot) = g.in_flight.get_mut(&job.id) {
            *slot = record.clone();
        }
        job.content_hash = Some(record.content_hash.clone());
        job.updated_at_ms = now;
        if outcome.completed_stage == StageName::Receive {
            // Advance.
        }
        Ok(outcome.object_complete)
    }

    // ------------------------------------------------------------------
    // Read API (permissions + retention)
    // ------------------------------------------------------------------

    fn check_access(&self, rec: &ObjectRecord, reader: &Reader) -> Result<()> {
        let g = self.inner.lock();
        let now = g.now();
        let live = g.live_session.clone();
        drop(g);
        rec.permissions.check_read(reader)?;
        rec.retention
            .check_readable(now, live.as_deref())?;
        Ok(())
    }

    /// Look up object by content hash (metadata only).
    pub fn get_record(&self, hash: &ContentHash, reader: &Reader) -> Result<ObjectRecord> {
        let g = self.inner.lock();
        let rec = g
            .objects
            .get(hash.as_str())
            .cloned()
            .ok_or_else(|| ObjectError::NotFound(hash.as_str().into()))?;
        drop(g);
        self.check_access(&rec, reader)?;
        Ok(rec)
    }

    /// All refs pointing at a content hash.
    pub fn refs_for(&self, hash: &ContentHash) -> Vec<ObjectRef> {
        let g = self.inner.lock();
        g.refs
            .values()
            .filter(|r| r.content_hash == *hash)
            .cloned()
            .collect()
    }

    pub fn get_ref(&self, ref_id: &str) -> Result<ObjectRef> {
        let g = self.inner.lock();
        g.refs
            .get(ref_id)
            .cloned()
            .ok_or_else(|| ObjectError::RefNotFound(ref_id.into()))
    }

    /// Count of unique objects (by content hash).
    pub fn object_count(&self) -> usize {
        self.inner.lock().objects.len()
    }

    pub fn ref_count(&self) -> usize {
        self.inner.lock().refs.len()
    }

    /// Context-compile path: derivatives only. Never raw bytes.
    pub fn compile_view(
        &self,
        hash: &ContentHash,
        reader: &Reader,
        selection: &DerivativeSelection,
        label: Option<String>,
    ) -> Result<CompileObjectView> {
        let rec = self.get_record(hash, reader)?;
        rec.permissions.check_model_derivatives(reader)?;
        if rec.status != ObjectStatus::Ready {
            return Err(ObjectError::StageNotReady {
                stage: "finalize".into(),
                status: format!("{:?}", rec.status),
            });
        }
        Ok(CompileObjectView::from_record(
            &rec,
            &selection.kinds,
            label,
        ))
    }

    /// Compile view by ref id.
    pub fn compile_view_for_ref(
        &self,
        ref_id: &str,
        reader: &Reader,
        selection: &DerivativeSelection,
    ) -> Result<CompileObjectView> {
        let r = self.get_ref(ref_id)?;
        self.compile_view(&r.content_hash, reader, selection, r.label)
    }

    /// Raw body bytes — requires [`RawBytesCap`]. Not available on compile path.
    pub fn raw_bytes(
        &self,
        hash: &ContentHash,
        reader: &Reader,
        _cap: &RawBytesCap,
    ) -> Result<Vec<u8>> {
        let rec = self.get_record(hash, reader)?;
        if !rec.permissions.allow_export {
            return Err(ObjectError::PermissionDenied {
                reason: "allow_export is false".into(),
            });
        }
        match &rec.location {
            ObjectLocation::Local { path } => {
                let mut f = File::open(path)?;
                let mut out = Vec::new();
                // Still stream into vec for API simplicity; callers with large
                // files should use raw_reader. Cap via budget max_object.
                f.read_to_end(&mut out)?;
                let actual = ContentHash::of_bytes(&out);
                if actual != *hash {
                    return Err(ObjectError::ContentAddressMismatch {
                        expected: hash.as_str().into(),
                        actual: actual.as_str().into(),
                    });
                }
                Ok(out)
            }
            other => Err(ObjectError::Invalid(format!(
                "raw_bytes not available for location {other:?}"
            ))),
        }
    }

    /// Stream raw body into `writer` without requiring full RAM. Needs cap.
    pub fn raw_stream_to<W: Write>(
        &self,
        hash: &ContentHash,
        reader: &Reader,
        _cap: &RawBytesCap,
        writer: &mut W,
    ) -> Result<usize> {
        let rec = self.get_record(hash, reader)?;
        if !rec.permissions.allow_export {
            return Err(ObjectError::PermissionDenied {
                reason: "allow_export is false".into(),
            });
        }
        let path = match &rec.location {
            ObjectLocation::Local { path } => path.clone(),
            other => {
                return Err(ObjectError::Invalid(format!(
                    "raw stream not available for {other:?}"
                )))
            }
        };
        let mut f = File::open(path)?;
        let mut buf = vec![0u8; CHUNK_SIZE];
        let mut peak = 0usize;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            peak = peak.max(n);
            writer.write_all(&buf[..n])?;
        }
        Ok(peak)
    }

    /// Lookup job.
    pub fn job(&self, id: &str) -> Option<IngestJob> {
        self.inner.lock().queue.get(id).cloned()
    }

    /// Find object hash produced by a finished job (from in_flight merge / job).
    pub fn hash_for_job(&self, job_id: &str) -> Option<ContentHash> {
        let g = self.inner.lock();
        g.queue
            .get(job_id)
            .and_then(|j| j.content_hash.clone())
            .or_else(|| {
                g.in_flight
                    .get(job_id)
                    .map(|r| r.content_hash.clone())
            })
    }
}

fn write_bytes_chunked(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = File::create(path)?;
    let mut offset = 0;
    while offset < bytes.len() {
        let end = (offset + CHUNK_SIZE).min(bytes.len());
        f.write_all(&bytes[offset..end])?;
        offset = end;
    }
    f.flush()?;
    Ok(())
}

fn copy_chunked(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut input = File::open(src)?;
    let mut output = File::create(dst)?;
    let mut buf = vec![0u8; CHUNK_SIZE];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        output.write_all(&buf[..n])?;
    }
    output.flush()?;
    Ok(())
}
}

