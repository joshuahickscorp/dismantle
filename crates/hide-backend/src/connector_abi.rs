//! hide-connectors: HIDE YOU connector ABI and registry.
//!
//! YOU is the private personal-AI surface alongside CHAT and IDE. It reaches
//! the user's world through connectors — the highest-risk subsystem, because a
//! personal assistant touches real accounts.
//!
//! # Safety properties (the point)
//!
//! 1. **Default read-only, least privilege.** A connector that does not declare
//!    write has no [`ConnectorWrite`] impl; write is a type boundary.
//! 2. **No ambient credentials.** Every call takes an explicit [`AccountHandle`].
//!    There is no global credential lookup; one account cannot read another's.
//! 3. **Every write is an effect** that goes through [`PermissionGate`] and
//!    leaves a [`WriteReceipt`]. Silent execution is refused.
//! 4. **Connector data never silently enters global memory.** Ingest lands in
//!    connector scope; promotion to user/semantic requires an explicit cap.
//! 5. **Revocation is real.** Revoking an account invalidates handles; in-flight
//!    operations re-check and fail closed.
//!
//! # Implementation status
//!
//! Only `local_folder` and `rss` are implemented (fixture / local filesystem,
//! no network). Every other family is fully declared in the ABI and refuses
//! construction.
//!
//! ```
//! use crate::connector_abi::{AccountStore, ConnectorRegistry, CredentialMaterial, ListRequest};
//! use crate::connector_abi::abi::FamilyId;
//!
//! let reg = ConnectorRegistry::builtin();
//! assert!(reg.construct("github").is_err()); // declared, not constructible
//!
//! let folder = reg.construct("local_folder").unwrap();
//! let mut accounts = AccountStore::new();
//! // In real tests the root is a tempdir; doctest only shows the shape.
//! let _ = (folder, accounts, FamilyId::new("local_folder"), ListRequest::default(), CredentialMaterial { material: "/tmp".into() });
//! ```

pub use abi::{
    AuditPolicy, AuthMethod, ChangeTransport, ConnectorAbi, ConnectorScope, EffectClass, FamilyId,
    ImplementationStatus, ObjectType, OfflineCache, RateLimit, ReadCapability, RevocationPolicy,
    SyncMode, WriteCapability,
};
pub use account::{
    AccountHandle, AccountId, AccountStore, CredentialMaterial, InFlightGuard,
};
pub use connector::{
    BTreeMapStr, Connector, ConnectorObject, ConnectorRead, ConnectorWrite, DeclaredConnector,
    ListRequest, ReadRequest,
};
pub use effects::{
    execute_with_receipt, execute_without_receipt, ConnectorWriteProposal, PermissionDecision,
    PermissionGate, PermissionPolicy, WriteKind, WriteReceipt, WriteResult,
};
pub use error::{ConnectorError, Result};
pub use impls::{LocalFolderConnector, RssConnector};
pub use memory::{
    ConnectorIngestCap, ConnectorMemoryStore, MemoryRecord, MemoryScope, SemanticPromotionCap,
    UserMemoryPromotionCap,
};
pub use registry::{ConnectorRegistry, LiveConnector, RegistryDocument};

// --- inlined connector_abi/abi.rs ---
pub mod abi {
//! Connector ABI: the single declaration every YOU connector family files.
//!
//! Every family (implemented or merely declared) fills the same fields: read/
//! write capabilities, auth method, scopes, object types, sync mode, offline
//! cache, rate limits, effect classes, revocation, and audit receipts. A filled
//! ABI is not an implementation — only [`ImplementationStatus::Implemented`]
//! families are constructible.

use serde::{Deserialize, Serialize};

/// Stable family id string (e.g. `local_folder`, `rss`, `github`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FamilyId(pub String);

impl FamilyId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for FamilyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for FamilyId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for FamilyId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Whether this family has a real constructor or is ABI-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplementationStatus {
    /// End-to-end implementation exists (fixture-backed is fine; network is not required).
    Implemented,
    /// ABI filled; construction is refused.
    Declared,
}

impl ImplementationStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Implemented => "implemented",
            Self::Declared => "declared",
        }
    }
}

/// Read capability declaration. Default for every connector is at least list/
/// fetch of its object types; a family with `reads: false` is inert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadCapability {
    /// May list objects in scope.
    pub list: bool,
    /// May fetch object content/metadata.
    pub fetch: bool,
    /// May search within the connector's object space.
    pub search: bool,
}

impl ReadCapability {
    pub fn list_and_fetch() -> Self {
        Self {
            list: true,
            fetch: true,
            search: false,
        }
    }
    pub fn full() -> Self {
        Self {
            list: true,
            fetch: true,
            search: true,
        }
    }
    pub fn none() -> Self {
        Self {
            list: false,
            fetch: false,
            search: false,
        }
    }
}

/// Write capability declaration. Absence (or all-false) means the connector is
/// read-only at the type boundary — there is no `ConnectorWrite` impl to call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteCapability {
    pub create: bool,
    pub update: bool,
    pub delete: bool,
}

impl WriteCapability {
    pub fn none() -> Self {
        Self {
            create: false,
            update: false,
            delete: false,
        }
    }
    pub fn full() -> Self {
        Self {
            create: true,
            update: true,
            delete: true,
        }
    }
    pub fn is_writable(&self) -> bool {
        self.create || self.update || self.delete
    }
}

/// How the connector authenticates. No ambient credentials: auth is always
/// bound to an explicit account handle minted by the account store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    /// No credentials; path/scope is the account (local folders, fixture RSS).
    None,
    /// OAuth2 (authorization code / device / etc.). Not implemented here.
    OAuth2 {
        authorization_url: String,
        token_url: String,
    },
    /// Personal access token or API key held by the account store.
    ApiToken,
    /// OS keychain / platform secret for local services.
    LocalSecret,
    /// MCP session negotiated out of band.
    McpSession,
    /// Generic bearer / custom scheme declared by the family.
    Custom(String),
}

/// A permission scope the connector may request or hold.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConnectorScope {
    pub name: String,
    pub description: String,
    /// Whether this scope implies write (elevated).
    pub write: bool,
}

impl ConnectorScope {
    pub fn read(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            write: false,
        }
    }
    pub fn write(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            write: true,
        }
    }
}

/// Object types the connector can surface.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectType {
    pub name: String,
    pub description: String,
}

impl ObjectType {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

/// Incremental sync support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncMode {
    /// Full re-list only.
    FullOnly,
    /// Cursor / page token incremental.
    Cursor,
    /// Timestamp / etag based.
    Timestamp,
    /// Provider-specific delta tokens.
    DeltaToken,
}

/// How the connector learns about remote changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeTransport {
    /// No push; caller polls.
    Polling { min_interval_secs: u64 },
    /// Webhook / push notifications (declared only; not implemented here).
    Webhook { verification: String },
    /// Both polling and webhooks supported.
    PollingAndWebhook {
        min_interval_secs: u64,
        verification: String,
    },
    /// Local filesystem watch (inotify / FSEvents style).
    LocalWatch,
    /// Not applicable (static fixture, one-shot read).
    None,
}

/// Offline cache policy declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfflineCache {
    pub supported: bool,
    /// Max bytes the connector may retain offline (0 = none).
    pub max_bytes: u64,
    /// Whether cache may hold content bodies or only metadata.
    pub bodies: bool,
}

impl OfflineCache {
    pub fn none() -> Self {
        Self {
            supported: false,
            max_bytes: 0,
            bodies: false,
        }
    }
    pub fn metadata(max_bytes: u64) -> Self {
        Self {
            supported: true,
            max_bytes,
            bodies: false,
        }
    }
    pub fn full(max_bytes: u64) -> Self {
        Self {
            supported: true,
            max_bytes,
            bodies: true,
        }
    }
}

/// Rate limit declaration (honest ceiling; not a live limiter).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub burst: u32,
    pub notes: String,
}

impl RateLimit {
    pub fn local() -> Self {
        Self {
            requests_per_minute: 600,
            burst: 60,
            notes: "local / fixture; no remote provider limit".into(),
        }
    }
    pub fn remote(rpm: u32, burst: u32, notes: impl Into<String>) -> Self {
        Self {
            requests_per_minute: rpm,
            burst,
            notes: notes.into(),
        }
    }
}

/// Effect classes a connector operation may produce. Writes are always an
/// elevated effect and must leave an audit receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectClass {
    Read,
    Write,
    Delete,
    ExternalMutation,
    SecretAccess,
    Network,
}

impl EffectClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Delete => "delete",
            Self::ExternalMutation => "external_mutation",
            Self::SecretAccess => "secret_access",
            Self::Network => "network",
        }
    }
    pub fn is_elevated(self) -> bool {
        !matches!(self, Self::Read)
    }
}

/// Revocation semantics. Real revocation invalidates handles and fails in-flight
/// work closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevocationPolicy {
    /// Handles are generation-checked; revoke bumps generation.
    pub invalidates_handles: bool,
    /// In-flight ops re-check handle before completing.
    pub fail_closed_in_flight: bool,
    /// Remote token revoke is attempted when the provider supports it.
    pub remote_revoke: bool,
    pub notes: String,
}

impl RevocationPolicy {
    pub fn real_local() -> Self {
        Self {
            invalidates_handles: true,
            fail_closed_in_flight: true,
            remote_revoke: false,
            notes: "local account store generation bump; fail closed".into(),
        }
    }
    pub fn real_with_remote() -> Self {
        Self {
            invalidates_handles: true,
            fail_closed_in_flight: true,
            remote_revoke: true,
            notes: "local generation bump plus remote token revoke when implemented".into(),
        }
    }
}

/// Audit receipt policy for connector operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditPolicy {
    /// Every write must leave a receipt.
    pub write_receipts_required: bool,
    /// Reads may optionally leave receipts (list/fetch).
    pub read_receipts: bool,
    pub notes: String,
}

impl AuditPolicy {
    pub fn writes_required() -> Self {
        Self {
            write_receipts_required: true,
            read_receipts: false,
            notes: "every write is an effect that leaves a receipt; no silent execution".into(),
        }
    }
}

/// The full connector ABI declaration for one family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorAbi {
    pub family_id: FamilyId,
    pub display_name: String,
    pub description: String,
    pub status: ImplementationStatus,
    pub read: ReadCapability,
    pub write: WriteCapability,
    pub auth: AuthMethod,
    pub scopes: Vec<ConnectorScope>,
    pub object_types: Vec<ObjectType>,
    pub sync: SyncMode,
    pub change_transport: ChangeTransport,
    pub offline_cache: OfflineCache,
    pub rate_limit: RateLimit,
    pub effect_classes: Vec<EffectClass>,
    pub revocation: RevocationPolicy,
    pub audit: AuditPolicy,
    /// Honest notes: what is real vs declared-only.
    pub honesty_notes: String,
}

impl ConnectorAbi {
    /// True when the ABI claims write capability.
    pub fn declares_write(&self) -> bool {
        self.write.is_writable()
    }

    /// True when construction is allowed.
    pub fn is_implemented(&self) -> bool {
        self.status == ImplementationStatus::Implemented
    }

    /// Validate internal consistency of the declaration.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errs = Vec::new();
        if self.family_id.0.is_empty() {
            errs.push("family_id empty".into());
        }
        if self.declares_write() && !self.effect_classes.contains(&EffectClass::Write) {
            errs.push(format!(
                "{}: write capability requires EffectClass::Write",
                self.family_id
            ));
        }
        if self.declares_write() && !self.audit.write_receipts_required {
            errs.push(format!(
                "{}: write capability requires audit.write_receipts_required",
                self.family_id
            ));
        }
        if !self.revocation.invalidates_handles {
            errs.push(format!(
                "{}: revocation must invalidate handles",
                self.family_id
            ));
        }
        if !self.revocation.fail_closed_in_flight {
            errs.push(format!(
                "{}: revocation must fail closed in-flight",
                self.family_id
            ));
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}
}


// --- inlined connector_abi/account.rs ---
pub mod account {
//! Account handles: explicit, non-ambient credentials.
//!
//! There is no global credential lookup. A connector receives an
//! [`AccountHandle`] that the account store minted for one family and one
//! account. Handles carry a generation; revoking an account bumps the
//! generation so every outstanding handle fails closed.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::connector_abi::abi::FamilyId;
use crate::connector_abi::error::{ConnectorError, Result};

/// Stable account identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AccountId(pub String);

impl AccountId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AccountId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque credential material. Never ambient: only reachable through a live
/// [`AccountHandle`] that the store validates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialMaterial {
    /// Opaque token / path / fixture key. Not a global secret lookup key.
    pub material: String,
}

/// An explicit account handle passed into every connector call.
///
/// Constructed only by [`AccountStore::mint_handle`]. Connectors do not look up
/// credentials themselves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountHandle {
    pub account_id: AccountId,
    pub family_id: FamilyId,
    /// Generation at mint time; must match the store's current generation.
    pub generation: u64,
    /// Opaque credential token bound to this account only.
    pub(crate) credential: CredentialMaterial,
}

impl AccountHandle {
    pub fn account_id(&self) -> &AccountId {
        &self.account_id
    }
    pub fn family_id(&self) -> &FamilyId {
        &self.family_id
    }
    /// Credential material for *this* account only. Still requires the store
    /// to validate the handle is live before use.
    pub fn credential_material(&self) -> &str {
        &self.credential.material
    }
}

struct AccountRecord {
    family_id: FamilyId,
    credential: CredentialMaterial,
    generation: u64,
    revoked: bool,
    label: String,
}

/// The sole place credentials live. Connectors never reach into this store
/// globally; they only use the handle they were given, re-validated here.
#[derive(Default)]
pub struct AccountStore {
    accounts: BTreeMap<AccountId, AccountRecord>,
    /// Monotonic id for auto-generated account ids.
    next: AtomicU64,
}

impl AccountStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an account for a family with explicit credential material.
    /// Returns the account id. Does not mint a handle — call [`mint_handle`].
    pub fn register(
        &mut self,
        family_id: FamilyId,
        label: impl Into<String>,
        credential: CredentialMaterial,
    ) -> AccountId {
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = AccountId::new(format!("{}-{}", family_id.as_str(), n));
        self.accounts.insert(
            id.clone(),
            AccountRecord {
                family_id,
                credential,
                generation: 1,
                revoked: false,
                label: label.into(),
            },
        );
        id
    }

    /// Mint an explicit handle. This is the only way a connector receives
    /// credentials. There is no ambient / process-global lookup.
    pub fn mint_handle(&self, account_id: &AccountId) -> Result<AccountHandle> {
        let rec = self
            .accounts
            .get(account_id)
            .ok_or_else(|| ConnectorError::AccountNotFound(account_id.clone()))?;
        if rec.revoked {
            return Err(ConnectorError::AccountRevoked(account_id.clone()));
        }
        Ok(AccountHandle {
            account_id: account_id.clone(),
            family_id: rec.family_id.clone(),
            generation: rec.generation,
            credential: rec.credential.clone(),
        })
    }

    /// Validate a handle is still live for the expected family. Call at the
    /// start of every operation and again before completing an in-flight write.
    pub fn validate(&self, handle: &AccountHandle, expected_family: &FamilyId) -> Result<()> {
        if &handle.family_id != expected_family {
            return Err(ConnectorError::AccountFamilyMismatch {
                handle: handle.family_id.clone(),
                connector: expected_family.clone(),
            });
        }
        let rec = self
            .accounts
            .get(&handle.account_id)
            .ok_or_else(|| ConnectorError::AccountNotFound(handle.account_id.clone()))?;
        if rec.revoked {
            return Err(ConnectorError::AccountRevoked(handle.account_id.clone()));
        }
        if rec.generation != handle.generation {
            return Err(ConnectorError::StaleHandle);
        }
        if rec.family_id != handle.family_id {
            return Err(ConnectorError::AccountFamilyMismatch {
                handle: handle.family_id.clone(),
                connector: rec.family_id.clone(),
            });
        }
        // Credential isolation: the handle's material must match this account only.
        if rec.credential != handle.credential {
            return Err(ConnectorError::CredentialIsolation(
                handle.account_id.clone(),
                handle.family_id.clone(),
            ));
        }
        Ok(())
    }

    /// Revoke an account. Invalidates all outstanding handles (generation bump)
    /// and marks the account revoked so minting fails closed.
    pub fn revoke(&mut self, account_id: &AccountId) -> Result<()> {
        let rec = self
            .accounts
            .get_mut(account_id)
            .ok_or_else(|| ConnectorError::AccountNotFound(account_id.clone()))?;
        rec.revoked = true;
        rec.generation = rec.generation.saturating_add(1);
        // Clear credential material so any leaked copy cannot be re-used via a
        // forged handle with matching generation (generation already mismatch).
        rec.credential = CredentialMaterial {
            material: String::new(),
        };
        Ok(())
    }

    pub fn is_revoked(&self, account_id: &AccountId) -> bool {
        self.accounts
            .get(account_id)
            .map(|r| r.revoked)
            .unwrap_or(true)
    }

    pub fn label(&self, account_id: &AccountId) -> Option<&str> {
        self.accounts.get(account_id).map(|r| r.label.as_str())
    }

    /// Deliberately absent: there is no way to look up "the" credential for a
    /// family without an account id. This method documents that law.
    pub fn ambient_lookup_forbidden() -> ConnectorError {
        ConnectorError::AmbientCredentialForbidden
    }
}

/// Guard that re-validates a handle before completing an in-flight operation.
///
/// Does not borrow the store for its lifetime, so the store may be mutated
/// (e.g. revoked) between [`begin`](Self::begin) and
/// [`complete`](Self::complete). Dropping without `complete` is fine; the
/// operation simply did not finish.
pub struct InFlightGuard {
    handle: AccountHandle,
    family: FamilyId,
}

impl InFlightGuard {
    pub fn begin(
        store: &AccountStore,
        handle: &AccountHandle,
        family: &FamilyId,
    ) -> Result<Self> {
        store.validate(handle, family)?;
        Ok(Self {
            handle: handle.clone(),
            family: family.clone(),
        })
    }

    /// Re-check the handle against the (possibly updated) store. On revocation
    /// mid-flight this returns [`ConnectorError::AccountRevoked`] /
    /// [`ConnectorError::StaleHandle`] and the caller must not complete.
    pub fn complete(self, store: &AccountStore) -> Result<AccountHandle> {
        store.validate(&self.handle, &self.family)?;
        Ok(self.handle)
    }

    pub fn handle(&self) -> &AccountHandle {
        &self.handle
    }
}
}


// --- inlined connector_abi/connector.rs ---
pub mod connector {
//! Connector trait surface with a read/write type boundary.
//!
//! - Every live connector implements [`ConnectorRead`].
//! - Only connectors that declare write implement [`ConnectorWrite`].
//! - A function that needs to write takes `T: ConnectorWrite`, so a read-only
//!   connector cannot be asked to write at compile time.
//! - Declared (non-implemented) families have no type that implements either
//!   trait; construction is refused by the registry.

use serde::{Deserialize, Serialize};

use crate::connector_abi::abi::{ConnectorAbi, FamilyId, WriteCapability};
use crate::connector_abi::account::{AccountHandle, AccountStore, InFlightGuard};
use crate::connector_abi::error::{ConnectorError, Result};

/// A listed or fetched object from a connector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorObject {
    pub id: String,
    pub object_type: String,
    pub title: String,
    /// Optional body / summary text.
    pub content: Option<String>,
    pub metadata: BTreeMapStr,
}

/// Simple ordered string map for metadata (serde-friendly).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BTreeMapStr(pub std::collections::BTreeMap<String, String>);

impl BTreeMapStr {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, k: impl Into<String>, v: impl Into<String>) {
        self.0.insert(k.into(), v.into());
    }
}

/// Read request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadRequest {
    /// Object id, path, feed item id, etc.
    pub locator: String,
}

/// List request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListRequest {
    /// Optional prefix / folder / query.
    pub prefix: Option<String>,
    pub limit: usize,
}

impl Default for ListRequest {
    fn default() -> Self {
        Self {
            prefix: None,
            limit: 100,
        }
    }
}

/// Shared identity every connector exposes.
pub trait Connector: Send + Sync {
    fn family_id(&self) -> &FamilyId;
    fn abi(&self) -> &ConnectorAbi;
}

/// Read surface. All implemented connectors implement this.
pub trait ConnectorRead: Connector {
    fn list(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ListRequest,
    ) -> Result<Vec<ConnectorObject>>;

    fn fetch(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ReadRequest,
    ) -> Result<ConnectorObject>;
}

/// Write surface. A type boundary: only connectors that declare write implement
/// this trait. Read-only connectors (local_folder, rss) do not, so they cannot
/// be passed to functions that require [`ConnectorWrite`].
///
/// Implementations prepare proposals only; execution goes through the
/// permission gate and a write receipt (see [`crate::connector_abi::effects`]).
pub trait ConnectorWrite: ConnectorRead {
    /// The declared write capability. Must be writable.
    fn write_capability(&self) -> &WriteCapability;

    /// Prepare a write proposal. Does not execute. Requires a live handle.
    fn prepare_write(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        kind: crate::connector_abi::effects::WriteKind,
        target: impl Into<String>,
        payload: impl Into<String>,
        summary: impl Into<String>,
    ) -> Result<crate::connector_abi::effects::ConnectorWriteProposal> {
        if !self.write_capability().is_writable() {
            return Err(ConnectorError::WriteNotDeclared(self.family_id().clone()));
        }
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let kind = kind;
        let effect = match kind {
            crate::connector_abi::effects::WriteKind::Delete => crate::connector_abi::abi::EffectClass::Delete,
            _ => crate::connector_abi::abi::EffectClass::Write,
        };
        let proposal = crate::connector_abi::effects::ConnectorWriteProposal {
            family_id: self.family_id().clone(),
            account_id: handle.account_id().clone(),
            kind,
            effect,
            summary: summary.into(),
            target: target.into(),
            payload: payload.into(),
        };
        guard.complete(store)?;
        Ok(proposal)
    }
}

/// Marker used by the registry: a family that is only declared has no
/// constructible type. Attempting to construct yields
/// [`ConnectorError::DeclaredNotConstructible`].
pub struct DeclaredConnector;

impl DeclaredConnector {
    pub fn try_construct(family_id: FamilyId) -> Result<Self> {
        Err(ConnectorError::DeclaredNotConstructible(family_id))
    }
}
}


// --- inlined connector_abi/effects.rs ---
pub mod effects {
//! Write effects, permission gate, and audit receipts.
//!
//! Every write is an effect. A connector never silently mutates the world: it
//! prepares a [`ConnectorWriteProposal`], the [`PermissionGate`] authorizes it
//! into a [`WriteReceipt`], and only then may [`execute_with_receipt`] run.
//! Reads do not require receipts.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::connector_abi::abi::{EffectClass, FamilyId};
use crate::connector_abi::account::{AccountHandle, AccountId};
use crate::connector_abi::error::{ConnectorError, Result};

/// Kind of write a connector proposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteKind {
    Create,
    Update,
    Delete,
}

impl WriteKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
}

/// A prepared, un-executed connector write. Carries no authority by itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorWriteProposal {
    pub family_id: FamilyId,
    pub account_id: AccountId,
    pub kind: WriteKind,
    pub effect: EffectClass,
    /// Human summary for the permission UI / audit log.
    pub summary: String,
    /// Opaque target locator (path, message id, ...).
    pub target: String,
    /// Opaque payload bytes as UTF-8 JSON or plain text for fixtures.
    pub payload: String,
}

/// Permission decision recorded before any write executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow,
    Deny,
}

/// A write receipt: proof that a proposal was authorized. Required to execute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteReceipt {
    pub id: String,
    pub proposal: ConnectorWriteProposal,
    pub decision: PermissionDecision,
    pub issued_at_ms: u64,
    /// Blake3 digest of the proposal fields for tamper evidence.
    pub digest: String,
    /// Whether this receipt has already been consumed by an execute call.
    pub consumed: bool,
}

impl WriteReceipt {
    pub fn is_allow(&self) -> bool {
        self.decision == PermissionDecision::Allow && !self.consumed
    }
}

/// Result of a successfully executed write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteResult {
    pub receipt_id: String,
    pub target: String,
    pub notes: String,
}

fn proposal_digest(p: &ConnectorWriteProposal) -> String {
    let mut h = blake3::Hasher::new();
    h.update(p.family_id.as_str().as_bytes());
    h.update(b"|");
    h.update(p.account_id.as_str().as_bytes());
    h.update(b"|");
    h.update(p.kind.as_str().as_bytes());
    h.update(b"|");
    h.update(p.effect.as_str().as_bytes());
    h.update(b"|");
    h.update(p.summary.as_bytes());
    h.update(b"|");
    h.update(p.target.as_bytes());
    h.update(b"|");
    h.update(p.payload.as_bytes());
    h.finalize().to_hex().to_string()
}

/// Policy for auto-allowing certain write kinds in tests. Production callers
/// set a deny-by-default gate and only allow after explicit user approval.
#[derive(Debug, Clone, Default)]
pub struct PermissionPolicy {
    /// When true, every proposal is denied unless listed in `allow_targets`.
    pub deny_by_default: bool,
    /// Explicitly allowed target locators (exact match).
    pub allow_targets: Vec<String>,
}

impl PermissionPolicy {
    pub fn deny_by_default() -> Self {
        Self {
            deny_by_default: true,
            allow_targets: Vec::new(),
        }
    }
    pub fn allow_all_for_tests() -> Self {
        Self {
            deny_by_default: false,
            allow_targets: Vec::new(),
        }
    }
    pub fn allow_target(mut self, target: impl Into<String>) -> Self {
        self.allow_targets.push(target.into());
        self
    }
}

/// Permission + receipt authority for connector writes.
pub struct PermissionGate {
    policy: PermissionPolicy,
    receipts: BTreeMap<String, WriteReceipt>,
    next: AtomicU64,
    clock_ms: AtomicU64,
}

impl PermissionGate {
    pub fn new(policy: PermissionPolicy) -> Self {
        Self {
            policy,
            receipts: BTreeMap::new(),
            next: AtomicU64::new(0),
            clock_ms: AtomicU64::new(1),
        }
    }

    /// Authorize a proposal. Returns a receipt; only `Allow` receipts may
    /// execute. Denied proposals still leave a receipt for audit.
    pub fn authorize(&mut self, proposal: ConnectorWriteProposal) -> Result<WriteReceipt> {
        if proposal.effect == EffectClass::Read {
            return Err(ConnectorError::InvalidRequest(
                "read is not a write effect".into(),
            ));
        }
        let allowed = if self.policy.deny_by_default {
            self.policy.allow_targets.iter().any(|t| t == &proposal.target)
        } else {
            true
        };
        let decision = if allowed {
            PermissionDecision::Allow
        } else {
            PermissionDecision::Deny
        };
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = format!("wr-{}", n);
        let issued_at_ms = self.clock_ms.fetch_add(1, Ordering::Relaxed);
        let digest = proposal_digest(&proposal);
        let receipt = WriteReceipt {
            id: id.clone(),
            proposal,
            decision,
            issued_at_ms,
            digest,
            consumed: false,
        };
        self.receipts.insert(id, receipt.clone());
        Ok(receipt)
    }

    /// Consume an allow receipt. Second consume fails. Deny receipts cannot
    /// be consumed for execution.
    pub fn consume(&mut self, receipt_id: &str) -> Result<WriteReceipt> {
        let r = self
            .receipts
            .get_mut(receipt_id)
            .ok_or_else(|| ConnectorError::InvalidWriteReceipt(receipt_id.into()))?;
        if r.consumed {
            return Err(ConnectorError::InvalidWriteReceipt(receipt_id.into()));
        }
        if r.decision != PermissionDecision::Allow {
            return Err(ConnectorError::WritePermissionDenied(r.proposal.summary.clone()));
        }
        // Re-check digest integrity.
        let expected = proposal_digest(&r.proposal);
        if expected != r.digest {
            return Err(ConnectorError::InvalidWriteReceipt(receipt_id.into()));
        }
        r.consumed = true;
        Ok(r.clone())
    }

    pub fn get(&self, receipt_id: &str) -> Option<&WriteReceipt> {
        self.receipts.get(receipt_id)
    }

    pub fn all_receipts(&self) -> impl Iterator<Item = &WriteReceipt> {
        self.receipts.values()
    }
}

/// Execute a write only when a valid allow receipt is presented. This is the
/// sole execute entry point — there is no path that mutates without a receipt.
pub fn execute_with_receipt<F>(
    gate: &mut PermissionGate,
    receipt_id: &str,
    handle: &AccountHandle,
    mut body: F,
) -> Result<WriteResult>
where
    F: FnMut(&ConnectorWriteProposal, &AccountHandle) -> Result<WriteResult>,
{
    let receipt = gate.consume(receipt_id)?;
    if receipt.proposal.account_id != handle.account_id {
        return Err(ConnectorError::InvalidWriteReceipt(receipt_id.into()));
    }
    let mut result = body(&receipt.proposal, handle)?;
    result.receipt_id = receipt.id;
    Ok(result)
}

/// Attempt to execute without a receipt — always fails. Exists so tests can
/// prove silent execution is impossible.
pub fn execute_without_receipt(_proposal: &ConnectorWriteProposal) -> Result<WriteResult> {
    Err(ConnectorError::WriteReceiptRequired)
}
}


// --- inlined connector_abi/error.rs ---
pub mod error {
//! Connector error types. Fail closed on revocation, missing capability, and
//! unauthorized memory promotion.

use thiserror::Error;

use crate::connector_abi::abi::FamilyId;
use crate::connector_abi::account::AccountId;
use crate::connector_abi::memory::MemoryScope;

/// Result alias for connector operations.
pub type Result<T> = std::result::Result<T, ConnectorError>;

/// Errors produced by the connector ABI, registry, and implementations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConnectorError {
    #[error("family `{0}` is declared but not implemented; construction refused")]
    DeclaredNotConstructible(FamilyId),

    #[error("family `{0}` is not registered")]
    UnknownFamily(FamilyId),

    #[error("family `{0}` does not declare write capability; write is a type boundary refusal")]
    WriteNotDeclared(FamilyId),

    #[error("account `{0}` is revoked; operation failed closed")]
    AccountRevoked(AccountId),

    #[error("account handle generation mismatch (revoked or stale); fail closed")]
    StaleHandle,

    #[error("account handle family mismatch: handle is `{handle}`, connector is `{connector}`")]
    AccountFamilyMismatch {
        handle: FamilyId,
        connector: FamilyId,
    },

    #[error("no ambient credential lookup: connector must receive an explicit AccountHandle")]
    AmbientCredentialForbidden,

    #[error("account `{0}` not found in account store")]
    AccountNotFound(AccountId),

    #[error("credential bound to account `{0}` is not readable by family `{1}`")]
    CredentialIsolation(AccountId, FamilyId),

    #[error("write refused: no permission grant for effect `{0}`")]
    WritePermissionDenied(String),

    #[error("write refused: silent execution forbidden; a WriteReceipt is required")]
    WriteReceiptRequired,

    #[error("write receipt `{0}` is invalid or already consumed")]
    InvalidWriteReceipt(String),

    #[error("memory write to scope `{target}` refused from connector scope; promotion required")]
    SilentMemoryPromotion {
        target: MemoryScope,
    },

    #[error("promotion to user memory requires UserMemoryPromotionCap")]
    UserPromotionCapRequired,

    #[error("io: {0}")]
    Io(String),

    #[error("parse: {0}")]
    Parse(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("rate limit: {0}")]
    RateLimit(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

impl From<std::io::Error> for ConnectorError {
    fn from(e: std::io::Error) -> Self {
        ConnectorError::Io(e.to_string())
    }
}
}


// --- inlined connector_abi/families.rs ---
pub mod families {
//! Every YOU connector family ABI declaration.
//!
//! Only `local_folder` and `rss` are [`ImplementationStatus::Implemented`].
//! All other families are fully declared (ABI filled) but not constructible.

use crate::connector_abi::abi::{
    AuditPolicy, AuthMethod, ChangeTransport, ConnectorAbi, EffectClass, FamilyId,
    ImplementationStatus, ObjectType, OfflineCache, RateLimit, ReadCapability, RevocationPolicy,
    SyncMode, WriteCapability, ConnectorScope,
};

fn base(
    id: &str,
    name: &str,
    description: &str,
    status: ImplementationStatus,
) -> ConnectorAbi {
    ConnectorAbi {
        family_id: FamilyId::new(id),
        display_name: name.into(),
        description: description.into(),
        status,
        read: ReadCapability::list_and_fetch(),
        write: WriteCapability::none(),
        auth: AuthMethod::None,
        scopes: vec![],
        object_types: vec![],
        sync: SyncMode::FullOnly,
        change_transport: ChangeTransport::None,
        offline_cache: OfflineCache::none(),
        rate_limit: RateLimit::local(),
        effect_classes: vec![EffectClass::Read],
        revocation: RevocationPolicy::real_local(),
        audit: AuditPolicy::writes_required(),
        honesty_notes: String::new(),
    }
}

/// `local_folder` — real, fixture-backed directory reader.
pub fn local_folder() -> ConnectorAbi {
    let mut a = base(
        "local_folder",
        "Local Folder",
        "Read a local directory tree under an explicit root path bound to the account handle.",
        ImplementationStatus::Implemented,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::none(); // type boundary: no ConnectorWrite impl
    a.auth = AuthMethod::None;
    a.scopes = vec![ConnectorScope::read(
        "folder.read",
        "Read files and list directories under the account root",
    )];
    a.object_types = vec![
        ObjectType::new("file", "A regular file"),
        ObjectType::new("directory", "A directory entry"),
    ];
    a.sync = SyncMode::FullOnly;
    a.change_transport = ChangeTransport::LocalWatch;
    a.offline_cache = OfflineCache::none();
    a.rate_limit = RateLimit::local();
    a.effect_classes = vec![EffectClass::Read];
    a.revocation = RevocationPolicy::real_local();
    a.audit = AuditPolicy::writes_required();
    a.honesty_notes =
        "IMPLEMENTED against local filesystem. Read-only. No network. Account credential is the root path."
            .into();
    a
}

/// `rss` — real, fixture-backed feed parser.
pub fn rss() -> ConnectorAbi {
    let mut a = base(
        "rss",
        "RSS / Atom Feed",
        "Parse an RSS or Atom feed from a local fixture path or (declared) remote URL.",
        ImplementationStatus::Implemented,
    );
    a.read = ReadCapability::list_and_fetch();
    a.write = WriteCapability::none();
    a.auth = AuthMethod::None;
    a.scopes = vec![ConnectorScope::read("feed.read", "Read feed items")];
    a.object_types = vec![
        ObjectType::new("feed", "Feed metadata"),
        ObjectType::new("item", "A feed item / entry"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 300,
    };
    a.offline_cache = OfflineCache::full(8 * 1024 * 1024);
    a.rate_limit = RateLimit::local();
    a.effect_classes = vec![EffectClass::Read];
    a.revocation = RevocationPolicy::real_local();
    a.audit = AuditPolicy::writes_required();
    a.honesty_notes =
        "IMPLEMENTED against committed fixture XML only. No network fetch in this crate."
            .into();
    a
}

pub fn github() -> ConnectorAbi {
    let mut a = base(
        "github",
        "GitHub",
        "Repositories, issues, PRs, and file contents via the GitHub API.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: false,
    };
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://github.com/login/oauth/authorize".into(),
        token_url: "https://github.com/login/oauth/access_token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("repo", "Read repository contents and metadata"),
        ConnectorScope::write("repo", "Open PRs, comment, push (elevated)"),
        ConnectorScope::read("read:user", "Read user profile"),
    ];
    a.object_types = vec![
        ObjectType::new("repository", "A GitHub repository"),
        ObjectType::new("issue", "An issue"),
        ObjectType::new("pull_request", "A pull request"),
        ObjectType::new("file", "A file blob in a repo"),
        ObjectType::new("comment", "An issue or PR comment"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 60,
        verification: "github-hmac-sha256".into(),
    };
    a.offline_cache = OfflineCache::metadata(64 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(5000, 100, "GitHub REST primary rate limit (authenticated)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn google_drive() -> ConnectorAbi {
    let mut a = base(
        "google_drive",
        "Google Drive",
        "Files and folders in Google Drive.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
        token_url: "https://oauth2.googleapis.com/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read(
            "drive.readonly",
            "Read Drive files",
        ),
        ConnectorScope::write("drive.file", "Create/update app-created files"),
    ];
    a.object_types = vec![
        ObjectType::new("file", "A Drive file"),
        ObjectType::new("folder", "A Drive folder"),
        ObjectType::new("revision", "A file revision"),
    ];
    a.sync = SyncMode::DeltaToken;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 60,
        verification: "google-push".into(),
    };
    a.offline_cache = OfflineCache::full(256 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(1000, 100, "Drive API per-user quota (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn icloud_drive() -> ConnectorAbi {
    let mut a = base(
        "icloud_drive",
        "iCloud Drive / Local File Provider",
        "iCloud Drive via local file-provider mount or CloudKit (declared).",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::LocalSecret;
    a.scopes = vec![
        ConnectorScope::read("icloud.read", "Read iCloud Drive paths"),
        ConnectorScope::write("icloud.write", "Write iCloud Drive paths"),
    ];
    a.object_types = vec![
        ObjectType::new("file", "A file"),
        ObjectType::new("directory", "A directory"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::LocalWatch;
    a.offline_cache = OfflineCache::full(512 * 1024 * 1024);
    a.rate_limit = RateLimit::local();
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::SecretAccess,
    ];
    a.revocation = RevocationPolicy::real_local();
    a.honesty_notes =
        "DECLARED only. Distinct from implemented local_folder; not constructible.".into();
    a
}

pub fn gmail() -> ConnectorAbi {
    let mut a = base(
        "gmail",
        "Gmail",
        "Gmail messages, threads, labels.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: true,
    };
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
        token_url: "https://oauth2.googleapis.com/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("gmail.readonly", "Read mail"),
        ConnectorScope::write("gmail.send", "Send mail"),
        ConnectorScope::write("gmail.modify", "Modify labels / trash"),
    ];
    a.object_types = vec![
        ObjectType::new("message", "An email message"),
        ObjectType::new("thread", "A message thread"),
        ObjectType::new("label", "A Gmail label"),
        ObjectType::new("attachment", "A message attachment"),
    ];
    a.sync = SyncMode::DeltaToken;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 30,
        verification: "google-push".into(),
    };
    a.offline_cache = OfflineCache::metadata(128 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(250, 25, "Gmail API quota (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn google_calendar() -> ConnectorAbi {
    let mut a = base(
        "google_calendar",
        "Google Calendar",
        "Calendars and events.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
        token_url: "https://oauth2.googleapis.com/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("calendar.readonly", "Read calendars and events"),
        ConnectorScope::write("calendar.events", "Create/update/delete events"),
    ];
    a.object_types = vec![
        ObjectType::new("calendar", "A calendar"),
        ObjectType::new("event", "A calendar event"),
    ];
    a.sync = SyncMode::DeltaToken;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 60,
    };
    a.offline_cache = OfflineCache::metadata(32 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(1000, 100, "Calendar API quota (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn google_contacts() -> ConnectorAbi {
    let mut a = base(
        "google_contacts",
        "Google Contacts",
        "People / contacts directory.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: true,
    };
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
        token_url: "https://oauth2.googleapis.com/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("contacts.readonly", "Read contacts"),
        ConnectorScope::write("contacts", "Mutate contacts"),
    ];
    a.object_types = vec![ObjectType::new("person", "A contact / person")];
    a.sync = SyncMode::DeltaToken;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 300,
    };
    a.offline_cache = OfflineCache::metadata(16 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(90, 10, "People API quota (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn slack() -> ConnectorAbi {
    let mut a = base(
        "slack",
        "Slack",
        "Channels, messages, files.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: true,
    };
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://slack.com/oauth/v2/authorize".into(),
        token_url: "https://slack.com/api/oauth.v2.access".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("channels:history", "Read channel history"),
        ConnectorScope::read("files:read", "Read files"),
        ConnectorScope::write("chat:write", "Post messages"),
    ];
    a.object_types = vec![
        ObjectType::new("channel", "A channel"),
        ObjectType::new("message", "A message"),
        ObjectType::new("file", "A shared file"),
        ObjectType::new("user", "A workspace user"),
    ];
    a.sync = SyncMode::Cursor;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 30,
        verification: "slack-signing-secret".into(),
    };
    a.offline_cache = OfflineCache::metadata(64 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(50, 10, "Slack tiered rate limits (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn notion() -> ConnectorAbi {
    let mut a = base(
        "notion",
        "Notion",
        "Pages, databases, blocks.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://api.notion.com/v1/oauth/authorize".into(),
        token_url: "https://api.notion.com/v1/oauth/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("notion.read", "Read pages and databases"),
        ConnectorScope::write("notion.write", "Create/update pages and blocks"),
    ];
    a.object_types = vec![
        ObjectType::new("page", "A Notion page"),
        ObjectType::new("database", "A database"),
        ObjectType::new("block", "A block"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 60,
    };
    a.offline_cache = OfflineCache::metadata(64 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(180, 3, "Notion ~3 rps average (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. No OAuth, no network, not constructible.".into();
    a
}

pub fn dropbox_onedrive() -> ConnectorAbi {
    let mut a = base(
        "dropbox_onedrive",
        "Dropbox / OneDrive",
        "Cloud file storage via Dropbox or Microsoft Graph (OneDrive).",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://www.dropbox.com/oauth2/authorize".into(),
        token_url: "https://api.dropboxapi.com/oauth2/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("files.metadata.read", "List and metadata"),
        ConnectorScope::read("files.content.read", "Download content"),
        ConnectorScope::write("files.content.write", "Upload / modify"),
    ];
    a.object_types = vec![
        ObjectType::new("file", "A cloud file"),
        ObjectType::new("folder", "A cloud folder"),
    ];
    a.sync = SyncMode::Cursor;
    a.change_transport = ChangeTransport::PollingAndWebhook {
        min_interval_secs: 60,
        verification: "provider-webhook".into(),
    };
    a.offline_cache = OfflineCache::full(512 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(600, 50, "Provider-dependent (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. Dual-provider family; no live API, not constructible.".into();
    a
}

pub fn browser_search() -> ConnectorAbi {
    let mut a = base(
        "browser_search",
        "Browser and Search",
        "Web page fetch and search results (declared; no live crawl here).",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::none();
    a.auth = AuthMethod::None;
    a.scopes = vec![
        ConnectorScope::read("web.fetch", "Fetch a URL"),
        ConnectorScope::read("web.search", "Run a search query"),
    ];
    a.object_types = vec![
        ObjectType::new("page", "A web page capture"),
        ObjectType::new("search_result", "A search hit"),
    ];
    a.sync = SyncMode::FullOnly;
    a.change_transport = ChangeTransport::None;
    a.offline_cache = OfflineCache::full(128 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(60, 10, "Search provider limits (declared)");
    a.effect_classes = vec![EffectClass::Read, EffectClass::Network];
    a.revocation = RevocationPolicy::real_local();
    a.honesty_notes =
        "DECLARED only. No network. Distinct from hide-browser crate integration.".into();
    a
}

pub fn generic_mcp() -> ConnectorAbi {
    let mut a = base(
        "generic_mcp",
        "Generic MCP",
        "Model Context Protocol server as a connector (tools/resources).",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: true,
        delete: true,
    };
    a.auth = AuthMethod::McpSession;
    a.scopes = vec![
        ConnectorScope::read("mcp.resources", "List and read MCP resources"),
        ConnectorScope::write("mcp.tools", "Invoke MCP tools (may mutate)"),
    ];
    a.object_types = vec![
        ObjectType::new("resource", "An MCP resource"),
        ObjectType::new("tool_result", "Result of an MCP tool call"),
        ObjectType::new("prompt", "An MCP prompt template"),
    ];
    a.sync = SyncMode::FullOnly;
    a.change_transport = ChangeTransport::None;
    a.offline_cache = OfflineCache::none();
    a.rate_limit = RateLimit::remote(120, 20, "Server-dependent (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::ExternalMutation,
        EffectClass::Network,
        EffectClass::SecretAccess,
    ];
    a.revocation = RevocationPolicy::real_local();
    a.honesty_notes = "DECLARED only. No MCP session runtime in this crate.".into();
    a
}

pub fn generic_oauth_api() -> ConnectorAbi {
    let mut a = base(
        "generic_oauth_api",
        "Generic OAuth / API",
        "Catch-all OAuth2 + REST connector template for user-configured APIs.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability::full();
    a.auth = AuthMethod::OAuth2 {
        authorization_url: "https://example.invalid/oauth/authorize".into(),
        token_url: "https://example.invalid/oauth/token".into(),
    };
    a.scopes = vec![
        ConnectorScope::read("api.read", "Read API resources"),
        ConnectorScope::write("api.write", "Mutate API resources"),
    ];
    a.object_types = vec![ObjectType::new("resource", "A generic API resource")];
    a.sync = SyncMode::Cursor;
    a.change_transport = ChangeTransport::Polling {
        min_interval_secs: 120,
    };
    a.offline_cache = OfflineCache::metadata(32 * 1024 * 1024);
    a.rate_limit = RateLimit::remote(60, 10, "User-configured (declared)");
    a.effect_classes = vec![
        EffectClass::Read,
        EffectClass::Write,
        EffectClass::Delete,
        EffectClass::Network,
        EffectClass::SecretAccess,
        EffectClass::ExternalMutation,
    ];
    a.revocation = RevocationPolicy::real_with_remote();
    a.honesty_notes = "DECLARED only. Template family; not constructible.".into();
    a
}

pub fn hawking_artifact_registry() -> ConnectorAbi {
    let mut a = base(
        "hawking_artifact_registry",
        "Hawking Artifact Registry",
        "Local Hawking/HIDE artifact registry: receipts, models metadata, sealed packs.",
        ImplementationStatus::Declared,
    );
    a.read = ReadCapability::full();
    a.write = WriteCapability {
        create: true,
        update: false,
        delete: false,
    };
    a.auth = AuthMethod::LocalSecret;
    a.scopes = vec![
        ConnectorScope::read("artifacts.read", "Read artifact metadata and bytes"),
        ConnectorScope::write("artifacts.publish", "Publish a new artifact receipt"),
    ];
    a.object_types = vec![
        ObjectType::new("artifact", "A registered artifact"),
        ObjectType::new("receipt", "A verification or publish receipt"),
        ObjectType::new("manifest", "An artifact manifest"),
    ];
    a.sync = SyncMode::Timestamp;
    a.change_transport = ChangeTransport::LocalWatch;
    a.offline_cache = OfflineCache::metadata(64 * 1024 * 1024);
    a.rate_limit = RateLimit::local();
    a.effect_classes = vec![EffectClass::Read, EffectClass::Write, EffectClass::SecretAccess];
    a.revocation = RevocationPolicy::real_local();
    a.honesty_notes =
        "DECLARED only in this crate. Local artifact store wiring is a separate concern.".into();
    a
}

/// Every family ABI, stable order.
pub fn all_families() -> Vec<ConnectorAbi> {
    vec![
        local_folder(),
        rss(),
        github(),
        google_drive(),
        icloud_drive(),
        gmail(),
        google_calendar(),
        google_contacts(),
        slack(),
        notion(),
        dropbox_onedrive(),
        browser_search(),
        generic_mcp(),
        generic_oauth_api(),
        hawking_artifact_registry(),
    ]
}

/// Family ids that are implemented end-to-end.
pub fn implemented_family_ids() -> &'static [&'static str] {
    &["local_folder", "rss"]
}
}


// --- inlined connector_abi/impls.rs ---
pub mod impls {
//! Real connector implementations (fixture-backed, no network).

pub use local_folder::LocalFolderConnector;
pub use rss::RssConnector;

// --- inlined impls/local_folder.rs ---
pub mod local_folder {
//! `local_folder` connector: read a directory under an account-bound root.
//!
//! Read-only. Does **not** implement [`crate::connector_abi::connector::ConnectorWrite`] — that
//! is the type boundary for safety property 1.

use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::connector_abi::abi::{ConnectorAbi, FamilyId};
use crate::connector_abi::account::{AccountHandle, AccountStore, InFlightGuard};
use crate::connector_abi::connector::{
    BTreeMapStr, Connector, ConnectorObject, ConnectorRead, ListRequest, ReadRequest,
};
use crate::connector_abi::error::{ConnectorError, Result};
use crate::connector_abi::families;

/// Live local-folder connector. Account credential material is the absolute
/// root path the account may read.
#[derive(Debug)]
pub struct LocalFolderConnector {
    abi: ConnectorAbi,
}

impl LocalFolderConnector {
    pub fn new() -> Self {
        Self {
            abi: families::local_folder(),
        }
    }

    fn root_from_handle(&self, handle: &AccountHandle) -> Result<PathBuf> {
        let raw = handle.credential_material();
        if raw.is_empty() {
            return Err(ConnectorError::InvalidRequest(
                "local_folder account root path is empty".into(),
            ));
        }
        let p = PathBuf::from(raw);
        if !p.is_absolute() {
            return Err(ConnectorError::InvalidRequest(format!(
                "local_folder root must be absolute, got {raw}"
            )));
        }
        Ok(p)
    }

    /// Resolve a locator under root without allowing `..` escape.
    fn resolve(&self, root: &Path, locator: &str) -> Result<PathBuf> {
        let rel = Path::new(locator.trim_start_matches('/'));
        for c in rel.components() {
            match c {
                Component::Normal(_) | Component::CurDir => {}
                Component::ParentDir => {
                    return Err(ConnectorError::InvalidRequest(
                        "path escape (..) forbidden".into(),
                    ));
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(ConnectorError::InvalidRequest(
                        "absolute locator forbidden; use root-relative paths".into(),
                    ));
                }
            }
        }
        let joined = root.join(rel);
        // Canonicalize when the path exists; otherwise keep joined for not-found.
        if joined.exists() {
            let canon = joined.canonicalize().map_err(ConnectorError::from)?;
            let root_canon = root.canonicalize().map_err(ConnectorError::from)?;
            if !canon.starts_with(&root_canon) {
                return Err(ConnectorError::InvalidRequest(
                    "resolved path escapes account root".into(),
                ));
            }
            Ok(canon)
        } else {
            Ok(joined)
        }
    }

    fn entry_to_object(&self, root: &Path, path: &Path) -> Result<ConnectorObject> {
        let meta = fs::metadata(path).map_err(ConnectorError::from)?;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let id = if rel.is_empty() {
            ".".to_string()
        } else {
            rel
        };
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| id.clone());
        let object_type = if meta.is_dir() { "directory" } else { "file" };
        let mut metadata = BTreeMapStr::new();
        metadata.insert("size", meta.len().to_string());
        metadata.insert("is_dir", meta.is_dir().to_string());
        Ok(ConnectorObject {
            id,
            object_type: object_type.into(),
            title: name,
            content: None,
            metadata,
        })
    }
}

impl Default for LocalFolderConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for LocalFolderConnector {
    fn family_id(&self) -> &FamilyId {
        &self.abi.family_id
    }
    fn abi(&self) -> &ConnectorAbi {
        &self.abi
    }
}

impl ConnectorRead for LocalFolderConnector {
    fn list(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ListRequest,
    ) -> Result<Vec<ConnectorObject>> {
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let root = self.root_from_handle(handle)?;
        let dir = match &request.prefix {
            Some(p) if !p.is_empty() && p != "." => self.resolve(&root, p)?,
            _ => root.canonicalize().map_err(ConnectorError::from)?,
        };
        if !dir.is_dir() {
            return Err(ConnectorError::NotFound(format!(
                "not a directory: {}",
                dir.display()
            )));
        }
        let root_canon = root.canonicalize().map_err(ConnectorError::from)?;
        let mut out = Vec::new();
        let rd = fs::read_dir(&dir).map_err(ConnectorError::from)?;
        for ent in rd {
            let ent = ent.map_err(ConnectorError::from)?;
            let path = ent.path();
            out.push(self.entry_to_object(&root_canon, &path)?);
            if out.len() >= request.limit {
                break;
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        // Fail closed if revoked mid-flight before returning results.
        guard.complete(store)?;
        Ok(out)
    }

    fn fetch(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ReadRequest,
    ) -> Result<ConnectorObject> {
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let root = self.root_from_handle(handle)?;
        let root_canon = root.canonicalize().map_err(ConnectorError::from)?;
        let path = self.resolve(&root, &request.locator)?;
        if !path.exists() {
            return Err(ConnectorError::NotFound(request.locator.clone()));
        }
        let mut obj = self.entry_to_object(&root_canon, &path)?;
        if path.is_file() {
            // Read text when UTF-8; otherwise leave content empty and note binary.
            match fs::read_to_string(&path) {
                Ok(s) => obj.content = Some(s),
                Err(_) => {
                    obj.metadata.insert("binary", "true");
                }
            }
        }
        guard.complete(store)?;
        Ok(obj)
    }
}

// Deliberately no `impl ConnectorWrite for LocalFolderConnector`.
// That absence is the type boundary for default read-only / least privilege.
}


// --- inlined impls/rss.rs ---
pub mod rss {
//! `rss` connector: parse a committed RSS 2.0 fixture (no network).
//!
//! Read-only. Does **not** implement [`crate::connector_abi::connector::ConnectorWrite`].

use std::fs;
use std::path::PathBuf;

use crate::connector_abi::abi::{ConnectorAbi, FamilyId};
use crate::connector_abi::account::{AccountHandle, AccountStore, InFlightGuard};
use crate::connector_abi::connector::{
    BTreeMapStr, Connector, ConnectorObject, ConnectorRead, ListRequest, ReadRequest,
};
use crate::connector_abi::error::{ConnectorError, Result};
use crate::connector_abi::families;

/// One parsed feed item.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FeedItem {
    guid: String,
    title: String,
    link: String,
    description: String,
    pub_date: String,
}

/// Parsed feed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Feed {
    title: String,
    link: String,
    description: String,
    items: Vec<FeedItem>,
}

/// Minimal RSS 2.0 parser for committed fixtures. Not a general XML library —
/// deliberately small and network-free.
fn parse_rss_2(xml: &str) -> Result<Feed> {
    // Require channel.
    let channel = between(xml, "<channel>", "</channel>").ok_or_else(|| {
        ConnectorError::Parse("missing <channel> in RSS fixture".into())
    })?;
    let title = text_tag(channel, "title").unwrap_or_default();
    let link = text_tag(channel, "link").unwrap_or_default();
    let description = text_tag(channel, "description").unwrap_or_default();

    let mut items = Vec::new();
    let mut rest = channel;
    while let Some(item_body) = between(rest, "<item>", "</item>") {
        let item_title = text_tag(item_body, "title").unwrap_or_default();
        let item_link = text_tag(item_body, "link").unwrap_or_default();
        let item_desc = text_tag(item_body, "description").unwrap_or_default();
        let pub_date = text_tag(item_body, "pubDate").unwrap_or_default();
        let guid = text_tag(item_body, "guid")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                if !item_link.is_empty() {
                    item_link.clone()
                } else {
                    format!("item-{}", items.len())
                }
            });
        items.push(FeedItem {
            guid,
            title: item_title,
            link: item_link,
            description: item_desc,
            pub_date,
        });
        // Advance past this item.
        if let Some(idx) = rest.find("</item>") {
            rest = &rest[idx + "</item>".len()..];
        } else {
            break;
        }
    }
    Ok(Feed {
        title,
        link,
        description,
        items,
    })
}

fn between<'a>(s: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = s.find(open)? + open.len();
    let end = s[start..].find(close)? + start;
    Some(&s[start..end])
}

/// First direct-ish text tag content. Handles optional CDATA.
fn text_tag(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let mut search = s;
    while let Some(idx) = search.find(&open) {
        let after = &search[idx + open.len()..];
        // Skip attributes to '>'
        let gt = after.find('>')?;
        let body_start = gt + 1;
        // Self-closing?
        if after[..gt].ends_with('/') {
            search = &after[body_start..];
            continue;
        }
        let body = &after[body_start..];
        let end = body.find(&close)?;
        let raw = body[..end].trim();
        let text = if let Some(inner) = between(raw, "<![CDATA[", "]]>") {
            inner.trim().to_string()
        } else {
            decode_entities(raw)
        };
        return Some(text);
    }
    None
}

fn decode_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Live RSS connector. Account credential material is the absolute path to a
/// feed XML file (fixture).
#[derive(Debug)]
pub struct RssConnector {
    abi: ConnectorAbi,
}

impl RssConnector {
    pub fn new() -> Self {
        Self {
            abi: families::rss(),
        }
    }

    fn load_feed(&self, handle: &AccountHandle) -> Result<Feed> {
        let path = PathBuf::from(handle.credential_material());
        if path.as_os_str().is_empty() {
            return Err(ConnectorError::InvalidRequest(
                "rss account feed path is empty".into(),
            ));
        }
        let xml = fs::read_to_string(&path).map_err(|e| {
            ConnectorError::Io(format!("read feed {}: {e}", path.display()))
        })?;
        parse_rss_2(&xml)
    }
}

impl Default for RssConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for RssConnector {
    fn family_id(&self) -> &FamilyId {
        &self.abi.family_id
    }
    fn abi(&self) -> &ConnectorAbi {
        &self.abi
    }
}

impl ConnectorRead for RssConnector {
    fn list(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ListRequest,
    ) -> Result<Vec<ConnectorObject>> {
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let feed = self.load_feed(handle)?;
        let mut out = Vec::new();
        // Feed itself as first object when no prefix filter.
        if request.prefix.is_none() {
            let mut meta = BTreeMapStr::new();
            meta.insert("link", feed.link.clone());
            out.push(ConnectorObject {
                id: "feed".into(),
                object_type: "feed".into(),
                title: feed.title.clone(),
                content: Some(feed.description.clone()),
                metadata: meta,
            });
        }
        for item in feed.items {
            if let Some(pref) = &request.prefix {
                if !item.guid.contains(pref.as_str()) && !item.title.contains(pref.as_str()) {
                    continue;
                }
            }
            let mut meta = BTreeMapStr::new();
            meta.insert("link", item.link);
            meta.insert("pub_date", item.pub_date);
            out.push(ConnectorObject {
                id: item.guid,
                object_type: "item".into(),
                title: item.title,
                content: Some(item.description),
                metadata: meta,
            });
            if out.len() >= request.limit {
                break;
            }
        }
        guard.complete(store)?;
        Ok(out)
    }

    fn fetch(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ReadRequest,
    ) -> Result<ConnectorObject> {
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let feed = self.load_feed(handle)?;
        if request.locator == "feed" || request.locator.is_empty() {
            let mut meta = BTreeMapStr::new();
            meta.insert("link", feed.link);
            let obj = ConnectorObject {
                id: "feed".into(),
                object_type: "feed".into(),
                title: feed.title,
                content: Some(feed.description),
                metadata: meta,
            };
            guard.complete(store)?;
            return Ok(obj);
        }
        for item in feed.items {
            if item.guid == request.locator || item.link == request.locator {
                let mut meta = BTreeMapStr::new();
                meta.insert("link", item.link);
                meta.insert("pub_date", item.pub_date);
                let obj = ConnectorObject {
                    id: item.guid,
                    object_type: "item".into(),
                    title: item.title,
                    content: Some(item.description),
                    metadata: meta,
                };
                guard.complete(store)?;
                return Ok(obj);
            }
        }
        Err(ConnectorError::NotFound(request.locator.clone()))
    }
}

// Deliberately no `impl ConnectorWrite for RssConnector`.

#[cfg(test)]
mod parse_tests {
    use super::*;
    #[test]
    fn parses_minimal_rss() {
        let xml = r#"<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>T</title><link>http://example.test/</link><description>D</description>
<item>
<title>Item One</title>
<link>http://example.test/1</link>
<guid>guid-1</guid>
<description>Hello</description>
<pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
</item>
</channel></rss>"#;
        let feed = parse_rss_2(xml).unwrap();
        assert_eq!(feed.title, "T");
        assert_eq!(feed.items.len(), 1);
        assert_eq!(feed.items[0].guid, "guid-1");
        assert_eq!(feed.items[0].title, "Item One");
    }
}
}

}


// --- inlined connector_abi/memory.rs ---
pub mod memory {
//! Connector-scoped memory. Ingested connector content never silently enters
//! global (`user` / `semantic`) memory.
//!
//! Connector reads land in [`MemoryScope::Connector`]. Promotion to
//! [`MemoryScope::User`] or [`MemoryScope::Semantic`] requires an explicit
//! capability mint ([`UserMemoryPromotionCap`] / [`SemanticPromotionCap`]).
//! The connector read path never holds those caps.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::connector_abi::abi::FamilyId;
use crate::connector_abi::account::AccountId;
use crate::connector_abi::error::{ConnectorError, Result};

/// Memory scope for connector-ingested content.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// Content scoped to one connector account. Default landing zone.
    Connector {
        family_id: FamilyId,
        account_id: AccountId,
    },
    /// Project / semantic memory. Requires explicit promotion.
    Semantic,
    /// User preference / standing memory. Requires explicit promotion.
    User,
}

impl MemoryScope {
    pub fn connector(family_id: FamilyId, account_id: AccountId) -> Self {
        Self::Connector {
            family_id,
            account_id,
        }
    }
    pub fn as_label(&self) -> String {
        match self {
            Self::Connector {
                family_id,
                account_id,
            } => format!("connector:{}:{}", family_id, account_id),
            Self::Semantic => "semantic".into(),
            Self::User => "user".into(),
        }
    }
}

impl std::fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_label())
    }
}

/// One memory record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub scope: MemoryScope,
    pub content: String,
    pub source_object_id: String,
    pub written_at_ms: u64,
}

/// Capability: promote connector content into user memory.
/// Mint only at the explicit user-intent entry point. Connector read paths
/// must not hold this type.
#[derive(Debug, Clone, Copy)]
pub struct UserMemoryPromotionCap {
    _private: (),
}

impl UserMemoryPromotionCap {
    /// Mint only at the user-intent entry point.
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// Capability: promote connector content into semantic memory.
#[derive(Debug, Clone, Copy)]
pub struct SemanticPromotionCap {
    _private: (),
}

impl SemanticPromotionCap {
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// Capability held by connector read/ingest paths. Can only write connector scope.
#[derive(Debug, Clone, Copy)]
pub struct ConnectorIngestCap {
    _private: (),
}

impl ConnectorIngestCap {
    /// Mint for a connector ingest path. Does not authorize user/semantic writes.
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// In-memory store used to prove scope isolation. Not the production memory
/// system — just the boundary for connector content.
#[derive(Default)]
pub struct ConnectorMemoryStore {
    records: BTreeMap<String, MemoryRecord>,
    next: AtomicU64,
    clock_ms: AtomicU64,
}

impl ConnectorMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest connector content into connector scope only.
    ///
    /// Type boundary: takes [`ConnectorIngestCap`], not a user/semantic cap.
    /// There is no overload that writes user memory from this path.
    pub fn ingest_connector(
        &mut self,
        _cap: &ConnectorIngestCap,
        family_id: FamilyId,
        account_id: AccountId,
        source_object_id: impl Into<String>,
        content: impl Into<String>,
    ) -> MemoryRecord {
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = format!("cm-{}", n);
        let rec = MemoryRecord {
            id: id.clone(),
            scope: MemoryScope::connector(family_id, account_id),
            content: content.into(),
            source_object_id: source_object_id.into(),
            written_at_ms: self.clock_ms.fetch_add(1, Ordering::Relaxed),
        };
        self.records.insert(id, rec.clone());
        rec
    }

    /// Attempt to write user memory from a connector ingest path.
    ///
    /// Always refuses. Exists so the safety test can name the property:
    /// a connector read cannot write `user` memory.
    pub fn ingest_as_user_from_connector(
        &mut self,
        _cap: &ConnectorIngestCap,
        _content: impl Into<String>,
    ) -> Result<MemoryRecord> {
        Err(ConnectorError::SilentMemoryPromotion {
            target: MemoryScope::User,
        })
    }

    /// Explicit promotion to user memory. Requires [`UserMemoryPromotionCap`].
    pub fn promote_to_user(
        &mut self,
        _cap: &UserMemoryPromotionCap,
        record_id: &str,
    ) -> Result<MemoryRecord> {
        let src = self
            .records
            .get(record_id)
            .ok_or_else(|| ConnectorError::NotFound(record_id.into()))?
            .clone();
        if !matches!(src.scope, MemoryScope::Connector { .. }) {
            return Err(ConnectorError::InvalidRequest(
                "only connector-scoped records can be promoted".into(),
            ));
        }
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = format!("um-{}", n);
        let rec = MemoryRecord {
            id: id.clone(),
            scope: MemoryScope::User,
            content: src.content,
            source_object_id: src.id,
            written_at_ms: self.clock_ms.fetch_add(1, Ordering::Relaxed),
        };
        self.records.insert(id, rec.clone());
        Ok(rec)
    }

    /// Explicit promotion to semantic memory.
    pub fn promote_to_semantic(
        &mut self,
        _cap: &SemanticPromotionCap,
        record_id: &str,
    ) -> Result<MemoryRecord> {
        let src = self
            .records
            .get(record_id)
            .ok_or_else(|| ConnectorError::NotFound(record_id.into()))?
            .clone();
        if !matches!(src.scope, MemoryScope::Connector { .. }) {
            return Err(ConnectorError::InvalidRequest(
                "only connector-scoped records can be promoted".into(),
            ));
        }
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = format!("sm-{}", n);
        let rec = MemoryRecord {
            id: id.clone(),
            scope: MemoryScope::Semantic,
            content: src.content,
            source_object_id: src.id,
            written_at_ms: self.clock_ms.fetch_add(1, Ordering::Relaxed),
        };
        self.records.insert(id, rec.clone());
        Ok(rec)
    }

    pub fn get(&self, id: &str) -> Option<&MemoryRecord> {
        self.records.get(id)
    }

    pub fn in_scope(&self, scope: &MemoryScope) -> Vec<&MemoryRecord> {
        self.records
            .values()
            .filter(|r| &r.scope == scope)
            .collect()
    }

    pub fn user_records(&self) -> Vec<&MemoryRecord> {
        self.records
            .values()
            .filter(|r| matches!(r.scope, MemoryScope::User))
            .collect()
    }
}
}


// --- inlined connector_abi/registry.rs ---
pub mod registry {
//! Connector registry: every family ABI, construct only the implemented ones.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::connector_abi::abi::{ConnectorAbi, FamilyId, ImplementationStatus};
use crate::connector_abi::connector::{Connector, ConnectorRead, DeclaredConnector};
use crate::connector_abi::error::{ConnectorError, Result};
use crate::connector_abi::families;
use crate::connector_abi::impls::{LocalFolderConnector, RssConnector};

/// What the registry can hand back as a live connector.
#[derive(Debug)]
pub enum LiveConnector {
    LocalFolder(LocalFolderConnector),
    Rss(RssConnector),
}

impl LiveConnector {
    pub fn family_id(&self) -> &FamilyId {
        match self {
            Self::LocalFolder(c) => c.family_id(),
            Self::Rss(c) => c.family_id(),
        }
    }

    pub fn as_read(&self) -> &dyn ConnectorRead {
        match self {
            Self::LocalFolder(c) => c,
            Self::Rss(c) => c,
        }
    }
}

/// The YOU connector registry.
pub struct ConnectorRegistry {
    by_id: BTreeMap<String, ConnectorAbi>,
}

impl ConnectorRegistry {
    /// Built-in registry with every family declaration.
    pub fn builtin() -> Self {
        let mut by_id = BTreeMap::new();
        for abi in families::all_families() {
            by_id.insert(abi.family_id.as_str().to_string(), abi);
        }
        Self { by_id }
    }

    pub fn get(&self, id: &str) -> Option<&ConnectorAbi> {
        self.by_id.get(id)
    }

    pub fn families(&self) -> impl Iterator<Item = &ConnectorAbi> {
        self.by_id.values()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn implemented(&self) -> Vec<&ConnectorAbi> {
        self.families()
            .filter(|a| a.status == ImplementationStatus::Implemented)
            .collect()
    }

    pub fn declared(&self) -> Vec<&ConnectorAbi> {
        self.families()
            .filter(|a| a.status == ImplementationStatus::Declared)
            .collect()
    }

    /// Validate every ABI for internal consistency.
    pub fn validate_all(&self) -> std::result::Result<(), Vec<String>> {
        let mut errs = Vec::new();
        for a in self.families() {
            if let Err(e) = a.validate() {
                errs.extend(e);
            }
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }

    /// Construct a live connector. Declared families return
    /// [`ConnectorError::DeclaredNotConstructible`].
    pub fn construct(&self, family_id: &str) -> Result<LiveConnector> {
        let abi = self
            .by_id
            .get(family_id)
            .ok_or_else(|| ConnectorError::UnknownFamily(FamilyId::new(family_id)))?;
        match abi.status {
            ImplementationStatus::Declared => {
                // Explicit: declared connectors are not constructible.
                let _ = DeclaredConnector::try_construct(abi.family_id.clone());
                Err(ConnectorError::DeclaredNotConstructible(abi.family_id.clone()))
            }
            ImplementationStatus::Implemented => match family_id {
                "local_folder" => Ok(LiveConnector::LocalFolder(LocalFolderConnector::new())),
                "rss" => Ok(LiveConnector::Rss(RssConnector::new())),
                other => Err(ConnectorError::InvalidRequest(format!(
                    "family {other} marked implemented but has no constructor"
                ))),
            },
        }
    }

    /// Export the registry document for `HIDE_YOU_CONNECTOR_REGISTRY.json`.
    pub fn export_document(&self) -> RegistryDocument {
        let mut families: Vec<ConnectorAbi> = self.families().cloned().collect();
        families.sort_by(|a, b| a.family_id.as_str().cmp(b.family_id.as_str()));
        let implemented: Vec<String> = families
            .iter()
            .filter(|a| a.status == ImplementationStatus::Implemented)
            .map(|a| a.family_id.as_str().to_string())
            .collect();
        let declared: Vec<String> = families
            .iter()
            .filter(|a| a.status == ImplementationStatus::Declared)
            .map(|a| a.family_id.as_str().to_string())
            .collect();
        RegistryDocument {
            schema: "hide.you.connector_registry.v1".into(),
            surface: "YOU".into(),
            crate_name: "hide-connectors".into(),
            safety_properties: vec![
                "default_read_only_type_boundary".into(),
                "no_ambient_credentials".into(),
                "every_write_is_effect_with_receipt".into(),
                "connector_data_not_silent_global_memory".into(),
                "revocation_fail_closed".into(),
            ],
            implemented,
            declared,
            families,
            notes: vec![
                "Only local_folder and rss are constructible; all others refuse construction.".into(),
                "No real credentials, OAuth flows, or network calls in this crate.".into(),
                "Write-capable families declare write in ABI but have no ConnectorWrite impl until implemented.".into(),
            ],
        }
    }

    /// Write the registry JSON to a path (deterministic pretty JSON).
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let doc = self.export_document();
        let text = serde_json::to_string_pretty(&doc)
            .map_err(|e| ConnectorError::Parse(e.to_string()))?;
        std::fs::write(path, text + "\n").map_err(ConnectorError::from)
    }
}

impl Default for ConnectorRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

/// Top-level document for `HIDE_YOU_CONNECTOR_REGISTRY.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryDocument {
    pub schema: String,
    pub surface: String,
    pub crate_name: String,
    pub safety_properties: Vec<String>,
    pub implemented: Vec<String>,
    pub declared: Vec<String>,
    pub families: Vec<ConnectorAbi>,
    pub notes: Vec<String>,
}
}

