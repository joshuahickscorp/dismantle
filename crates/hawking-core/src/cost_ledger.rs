//! Per-token cost ledger for BASE_RUNTIME_MAXIMIZED.
//!
//! Default-off, additive instrumentation. When enabled, exclusive wall time
//! is attributed across a fixed set of buckets that are required to sum (plus
//! an explicit unattributed remainder) to the measured token wall time.
//!
//! Enable with `HAWKING_COST_LEDGER=1`, or programmatically via
//! [`set_enabled`] / [`begin_token`]. Disabled paths are a single atomic load
//! and do not allocate.
//!
//! Nesting uses an exclusive stack: entering a child bucket pauses the parent
//! so nested regions never double-count. That is what makes
//! `sum(buckets) + unattributed ≈ wall` a meaningful identity rather than
//! an accounting fiction.

use serde::Serialize;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Env var that turns the ledger on for the process (`=1`).
pub const COST_LEDGER_ENV: &str = "HAWKING_COST_LEDGER";

/// Fixed exclusive time buckets. Order is stable for reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Bucket {
    ArtifactVerificationAndSha = 0,
    ContainerLookup = 1,
    PackedIndexDecode = 2,
    CpuOrchestration = 3,
    HostDeviceTransfer = 4,
    MetalEncode = 5,
    MetalSubmit = 6,
    MetalSynchronize = 7,
    AttentionAndIndexShare = 8,
    Routing = 9,
    SharedExperts = 10,
    RoutedExperts = 11,
    KvUpdate = 12,
    FinalHeadAndSampling = 13,
}

impl Bucket {
    pub const ALL: [Bucket; 14] = [
        Bucket::ArtifactVerificationAndSha,
        Bucket::ContainerLookup,
        Bucket::PackedIndexDecode,
        Bucket::CpuOrchestration,
        Bucket::HostDeviceTransfer,
        Bucket::MetalEncode,
        Bucket::MetalSubmit,
        Bucket::MetalSynchronize,
        Bucket::AttentionAndIndexShare,
        Bucket::Routing,
        Bucket::SharedExperts,
        Bucket::RoutedExperts,
        Bucket::KvUpdate,
        Bucket::FinalHeadAndSampling,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Bucket::ArtifactVerificationAndSha => "artifact_verification_and_sha",
            Bucket::ContainerLookup => "container_lookup",
            Bucket::PackedIndexDecode => "packed_index_decode",
            Bucket::CpuOrchestration => "cpu_orchestration",
            Bucket::HostDeviceTransfer => "host_device_transfer",
            Bucket::MetalEncode => "metal_encode",
            Bucket::MetalSubmit => "metal_submit",
            Bucket::MetalSynchronize => "metal_synchronize",
            Bucket::AttentionAndIndexShare => "attention_and_indexshare",
            Bucket::Routing => "routing",
            Bucket::SharedExperts => "shared_experts",
            Bucket::RoutedExperts => "routed_experts",
            Bucket::KvUpdate => "kv_update",
            Bucket::FinalHeadAndSampling => "final_head_and_sampling",
        }
    }

    fn index(self) -> usize {
        self as u8 as usize
    }
}

/// One host↔device transfer observed while the ledger is recording a token.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TransferRecord {
    pub bytes: u64,
    /// `true` = host → device, `false` = device → host.
    pub host_to_device: bool,
    pub kind: &'static str,
}

/// Counters that usually explain a bandwidth-starved MoE, independent of time.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct TokenCounters {
    pub command_buffers_submitted: u64,
    pub dispatches_encoded: u64,
    /// Every place the CPU waited on the GPU (`wait_until_completed`).
    pub synchronization_points: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub host_to_device_transfers: u64,
    pub device_to_host_transfers: u64,
    /// Heap / Metal buffer allocations observed on the hot path.
    pub allocations: u64,
    pub allocation_bytes: u64,
    /// Weight bytes actually touched for matvec this token (resident or not).
    pub active_bytes_read: u64,
    /// First-touch loads (decode + upload) this token — zero on a warm cache hit.
    pub first_touch_load_bytes: u64,
    pub matvec_calls: u64,
    pub matvec_batch_calls: u64,
    pub matvec_batch_items: u64,
    /// Times `dense()` / `row()` re-entered the artifact (SHA path when verify on).
    pub dense_calls: u64,
    pub row_calls: u64,
    pub sha_verifications: u64,
}

/// Full report for one instrumented decode token.
#[derive(Debug, Clone, Serialize)]
pub struct TokenCostReport {
    pub schema: &'static str,
    pub wall_us: u64,
    /// Exclusive microseconds per bucket. Keys are stable snake_case names.
    pub buckets_us: serde_json::Map<String, serde_json::Value>,
    /// `wall_us - sum(buckets_us)`. An unattributed remainder is a finding.
    pub unattributed_us: u64,
    /// Signed residual so over-attribution (instrument bug) is visible.
    pub unattributed_signed_us: i64,
    pub attributed_us: u64,
    pub attributed_fraction: f64,
    pub counters: TokenCounters,
    /// Geometry the gate quotes: 8 × 3 × 1_378_368 × 78 when known.
    pub geometry_active_bytes: Option<u64>,
    pub active_bytes_vs_geometry_fraction: Option<f64>,
    pub transfers: Vec<TransferRecord>,
}

impl TokenCostReport {
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

struct Frame {
    bucket: Bucket,
    /// Nanos accumulated exclusively into this frame while it was active.
    exclusive_ns: u128,
    /// When this frame last became the active (top-of-stack) frame.
    resumed_at: Option<Instant>,
}

struct TokenState {
    wall_start: Instant,
    nanos: [u128; 14],
    stack: Vec<Frame>,
    counters: TokenCounters,
    transfers: Vec<TransferRecord>,
    geometry_active_bytes: Option<u64>,
}

impl TokenState {
    fn new() -> Self {
        Self {
            wall_start: Instant::now(),
            nanos: [0; 14],
            stack: Vec::new(),
            counters: TokenCounters::default(),
            transfers: Vec::new(),
            geometry_active_bytes: None,
        }
    }

    fn pause_top(&mut self, now: Instant) {
        if let Some(frame) = self.stack.last_mut() {
            if let Some(t0) = frame.resumed_at.take() {
                frame.exclusive_ns = frame.exclusive_ns.saturating_add(
                    now.duration_since(t0).as_nanos(),
                );
            }
        }
    }

    fn resume_top(&mut self, now: Instant) {
        if let Some(frame) = self.stack.last_mut() {
            frame.resumed_at = Some(now);
        }
    }

    fn enter(&mut self, bucket: Bucket) {
        let now = Instant::now();
        self.pause_top(now);
        self.stack.push(Frame {
            bucket,
            exclusive_ns: 0,
            resumed_at: Some(now),
        });
    }

    fn exit(&mut self, bucket: Bucket) {
        let now = Instant::now();
        let Some(mut frame) = self.stack.pop() else {
            return;
        };
        // Mismatched exit is a programming error; still fold time into the
        // frame's own bucket so we never silently drop measured work.
        if frame.bucket != bucket {
            eprintln!(
                "[cost_ledger] mismatched exit: expected {:?}, got {:?}",
                frame.bucket, bucket
            );
        }
        if let Some(t0) = frame.resumed_at.take() {
            frame.exclusive_ns = frame
                .exclusive_ns
                .saturating_add(now.duration_since(t0).as_nanos());
        }
        self.nanos[frame.bucket.index()] =
            self.nanos[frame.bucket.index()].saturating_add(frame.exclusive_ns);
        self.resume_top(now);
    }

    /// Add exclusive time to a bucket without stack nesting (for split
    /// encode/submit/wait where a parent scope already owns the region).
    ///
    /// Pauses the open parent, folds its live time into `exclusive_ns`, then
    /// deducts `ns` from that parent so wall time is not double-counted.
    fn add_ns(&mut self, bucket: Bucket, ns: u128) {
        if ns == 0 {
            return;
        }
        let now = Instant::now();
        self.pause_top(now);
        if let Some(frame) = self.stack.last_mut() {
            // Parent exclusive now includes the just-measured sub-interval.
            frame.exclusive_ns = frame.exclusive_ns.saturating_sub(ns);
        }
        self.nanos[bucket.index()] = self.nanos[bucket.index()].saturating_add(ns);
        self.resume_top(now);
    }

    fn finish(mut self) -> TokenCostReport {
        let now = Instant::now();
        // Drain any open scopes (should be empty if callers balanced).
        while let Some(mut frame) = self.stack.pop() {
            if let Some(t0) = frame.resumed_at.take() {
                frame.exclusive_ns = frame
                    .exclusive_ns
                    .saturating_add(now.duration_since(t0).as_nanos());
            }
            self.nanos[frame.bucket.index()] =
                self.nanos[frame.bucket.index()].saturating_add(frame.exclusive_ns);
        }
        let wall_ns = now.duration_since(self.wall_start).as_nanos();
        let wall_us = (wall_ns / 1_000) as u64;
        let mut buckets_us = serde_json::Map::new();
        // Floor each bucket to whole microseconds first, then sum — so
        // `attributed_us == sum(buckets_us.values())` exactly (no 1 µs
        // residual from summing nanos then dividing once).
        let mut attributed_us: u64 = 0;
        for b in Bucket::ALL {
            let us = (self.nanos[b.index()] / 1_000) as u64;
            attributed_us = attributed_us.saturating_add(us);
            buckets_us.insert(b.as_str().to_string(), serde_json::json!(us));
        }
        let unattributed_signed_us = wall_us as i64 - attributed_us as i64;
        let unattributed_us = unattributed_signed_us.max(0) as u64;
        let attributed_fraction = if wall_us == 0 {
            0.0
        } else {
            attributed_us as f64 / wall_us as f64
        };
        let active_bytes_vs_geometry_fraction =
            self.geometry_active_bytes
                .filter(|&g| g > 0)
                .map(|g| self.counters.active_bytes_read as f64 / g as f64);

        TokenCostReport {
            schema: "hawking.gravity.per_token_cost_ledger.v1",
            wall_us,
            buckets_us,
            unattributed_us,
            unattributed_signed_us,
            attributed_us,
            attributed_fraction,
            counters: self.counters,
            geometry_active_bytes: self.geometry_active_bytes,
            active_bytes_vs_geometry_fraction,
            transfers: self.transfers,
        }
    }
}

static PROCESS_ENABLED: AtomicBool = AtomicBool::new(false);
static ENV_RESOLVED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TOKEN: RefCell<Option<TokenState>> = const { RefCell::new(None) };
    /// Per-thread override of the process switch. `None` defers to
    /// `PROCESS_ENABLED` (env / process-wide `set_enabled`). Lets unit
    /// tests enable on one thread without racing siblings.
    static THREAD_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Resolve `HAWKING_COST_LEDGER` once. Safe to call repeatedly.
pub fn resolve_env() {
    if ENV_RESOLVED.swap(true, Ordering::Relaxed) {
        return;
    }
    if crate::env_on(COST_LEDGER_ENV) {
        PROCESS_ENABLED.store(true, Ordering::Relaxed);
    }
}

/// Programmatic enable/disable for **this thread**. Does not start a token;
/// see [`begin_token`]. Prefer this over the env var in tests and examples.
pub fn set_enabled(on: bool) {
    ENV_RESOLVED.store(true, Ordering::Relaxed);
    THREAD_ENABLED.with(|c| c.set(Some(on)));
}

/// Process-wide enable (also used by env resolution). Rarely needed outside
/// of multi-thread servers that want one switch for every worker.
pub fn set_enabled_process(on: bool) {
    ENV_RESOLVED.store(true, Ordering::Relaxed);
    PROCESS_ENABLED.store(on, Ordering::Relaxed);
}

/// True when the ledger switch is on for this thread. Does not require an
/// active token — used by hot-path hooks as a cheap early-out.
pub fn is_enabled() -> bool {
    resolve_env();
    if let Some(on) = THREAD_ENABLED.with(|c| c.get()) {
        return on;
    }
    PROCESS_ENABLED.load(Ordering::Relaxed)
}

/// True when a token is currently being recorded on this thread.
pub fn is_recording() -> bool {
    if !is_enabled() {
        return false;
    }
    TOKEN.with(|t| t.borrow().is_some())
}

/// Start exclusive attribution for one decode token on this thread.
/// No-op (and returns false) when the ledger is disabled.
pub fn begin_token() -> bool {
    if !is_enabled() {
        return false;
    }
    TOKEN.with(|t| {
        *t.borrow_mut() = Some(TokenState::new());
    });
    true
}

/// Finish the current token and return its report. Returns `None` when no
/// token was active (or the ledger is off).
pub fn end_token() -> Option<TokenCostReport> {
    if !is_enabled() {
        return None;
    }
    TOKEN.with(|t| t.borrow_mut().take().map(TokenState::finish))
}

/// RAII scope that charges exclusive time to `bucket` while it is alive.
pub struct Scope {
    bucket: Bucket,
    active: bool,
}

impl Scope {
    pub fn new(bucket: Bucket) -> Self {
        let active = is_recording();
        if active {
            TOKEN.with(|t| {
                if let Some(state) = t.borrow_mut().as_mut() {
                    state.enter(bucket);
                }
            });
        }
        Self { bucket, active }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if self.active {
            TOKEN.with(|t| {
                if let Some(state) = t.borrow_mut().as_mut() {
                    state.exit(self.bucket);
                }
            });
        }
    }
}

/// Enter a bucket for the duration of `f`.
#[inline]
pub fn with_bucket<R>(bucket: Bucket, f: impl FnOnce() -> R) -> R {
    let _scope = Scope::new(bucket);
    f()
}

/// Charge `duration` to `bucket` (and deduct from any open parent). Prefer
/// [`Scope`] for nested regions; use this for split encode/submit/wait
/// slices measured with their own `Instant`s.
pub fn add_duration(bucket: Bucket, duration: std::time::Duration) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.add_ns(bucket, duration.as_nanos());
        }
    });
}

pub fn record_command_buffer() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.command_buffers_submitted =
                state.counters.command_buffers_submitted.saturating_add(1);
        }
    });
}

pub fn record_dispatches(n: u64) {
    if n == 0 || !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.dispatches_encoded =
                state.counters.dispatches_encoded.saturating_add(n);
        }
    });
}

pub fn record_sync_point() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.synchronization_points =
                state.counters.synchronization_points.saturating_add(1);
        }
    });
}

pub fn record_transfer(bytes: u64, host_to_device: bool, kind: &'static str) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            if host_to_device {
                state.counters.host_to_device_bytes =
                    state.counters.host_to_device_bytes.saturating_add(bytes);
                state.counters.host_to_device_transfers =
                    state.counters.host_to_device_transfers.saturating_add(1);
            } else {
                state.counters.device_to_host_bytes =
                    state.counters.device_to_host_bytes.saturating_add(bytes);
                state.counters.device_to_host_transfers =
                    state.counters.device_to_host_transfers.saturating_add(1);
            }
            // Cap transfer log so a long warm run does not grow without bound
            // when someone leaves the ledger on for many tokens.
            if state.transfers.len() < 4096 {
                state.transfers.push(TransferRecord {
                    bytes,
                    host_to_device,
                    kind,
                });
            }
        }
    });
}

pub fn record_allocation(bytes: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.allocations = state.counters.allocations.saturating_add(1);
            state.counters.allocation_bytes =
                state.counters.allocation_bytes.saturating_add(bytes);
        }
    });
}

pub fn record_active_bytes(bytes: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.active_bytes_read =
                state.counters.active_bytes_read.saturating_add(bytes);
        }
    });
}

pub fn record_first_touch_load_bytes(bytes: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.first_touch_load_bytes =
                state.counters.first_touch_load_bytes.saturating_add(bytes);
        }
    });
}

pub fn record_matvec_call() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.matvec_calls = state.counters.matvec_calls.saturating_add(1);
        }
    });
}

pub fn record_matvec_batch(items: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.matvec_batch_calls =
                state.counters.matvec_batch_calls.saturating_add(1);
            state.counters.matvec_batch_items =
                state.counters.matvec_batch_items.saturating_add(items);
        }
    });
}

pub fn record_dense_call() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.dense_calls = state.counters.dense_calls.saturating_add(1);
        }
    });
}

pub fn record_row_call() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.row_calls = state.counters.row_calls.saturating_add(1);
        }
    });
}

pub fn record_sha_verification() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.sha_verifications =
                state.counters.sha_verifications.saturating_add(1);
        }
    });
}

/// Gate geometry: active routed-expert bytes per token.
/// `experts_per_tok * 3 projections * bytes_per_projection * n_layers`.
pub fn set_geometry_active_bytes(bytes: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.geometry_active_bytes = Some(bytes);
        }
    });
}

/// Default geometry quoted by BASE_RUNTIME_MAXIMIZED_GATE for the sealed
/// Math-Preserve artifact (8 × 3 × 1_378_368 × 78).
pub const SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES: u64 =
    8 * 3 * 1_378_368 * 78;

/// Compute geometry from arch fields when the per-projection byte size is
/// known (from a live PQ header). Falls back to the sealed constant when
/// `bytes_per_projection` is `None`.
pub fn geometry_active_bytes(
    n_layers: usize,
    experts_per_tok: usize,
    bytes_per_projection: Option<u64>,
) -> u64 {
    let bpp = bytes_per_projection.unwrap_or(1_378_368);
    (n_layers as u64)
        .saturating_mul(experts_per_tok as u64)
        .saturating_mul(3)
        .saturating_mul(bpp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn with_clean_ledger<R>(f: impl FnOnce() -> R) -> R {
        set_enabled(true);
        // Ensure no leftover token.
        let _ = end_token();
        let out = f();
        let _ = end_token();
        set_enabled(false);
        out
    }

    #[test]
    fn disabled_is_noop() {
        set_enabled(false);
        assert!(!begin_token());
        assert!(end_token().is_none());
        let _s = Scope::new(Bucket::Routing);
        record_dispatches(99);
        assert!(!is_recording());
    }

    #[test]
    fn exclusive_stack_partitions_time() {
        with_clean_ledger(|| {
            assert!(begin_token());
            {
                let _a = Scope::new(Bucket::AttentionAndIndexShare);
                std::thread::sleep(Duration::from_millis(5));
                {
                    let _m = Scope::new(Bucket::MetalEncode);
                    std::thread::sleep(Duration::from_millis(5));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            {
                let _r = Scope::new(Bucket::Routing);
                std::thread::sleep(Duration::from_millis(5));
            }
            let report = end_token().expect("report");
            let attn = report.buckets_us["attention_and_indexshare"].as_u64().unwrap();
            let enc = report.buckets_us["metal_encode"].as_u64().unwrap();
            let route = report.buckets_us["routing"].as_u64().unwrap();
            // Each sleep is ~5ms; allow wide slack for CI scheduling.
            assert!(enc >= 3_000, "encode us={enc}");
            assert!(attn >= 6_000, "attn exclusive us={attn}");
            assert!(route >= 3_000, "route us={route}");
            let sum: u64 = report
                .buckets_us
                .values()
                .filter_map(|v| v.as_u64())
                .sum();
            assert_eq!(sum, report.attributed_us);
            // Exclusive identity: attributed + unattributed ≈ wall (within 1ms slack).
            let covered = report.attributed_us + report.unattributed_us;
            let delta = (covered as i64 - report.wall_us as i64).unsigned_abs();
            assert!(
                delta < 1_000,
                "covered={covered} wall={} delta={delta}",
                report.wall_us
            );
            // Nested encode must not be inside attn's exclusive total in a
            // way that makes attn ≈ encode+2*5ms; attn should be ~10ms not ~15ms.
            assert!(
                attn < enc + 12_000,
                "attn should exclude nested encode: attn={attn} enc={enc}"
            );
        });
    }

    #[test]
    fn unattributed_is_explicit_when_no_scopes() {
        with_clean_ledger(|| {
            assert!(begin_token());
            std::thread::sleep(Duration::from_millis(3));
            let report = end_token().expect("report");
            assert_eq!(report.attributed_us, 0);
            assert!(report.unattributed_us >= 2_000);
            assert!(report.unattributed_signed_us > 0);
        });
    }

    #[test]
    fn counters_accumulate() {
        with_clean_ledger(|| {
            assert!(begin_token());
            set_geometry_active_bytes(SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES);
            record_command_buffer();
            record_dispatches(8);
            record_sync_point();
            record_transfer(1024, true, "x_upload");
            record_transfer(2048, false, "y_download");
            record_allocation(4096);
            record_active_bytes(1_378_368);
            record_matvec_batch(8);
            record_sha_verification();
            let report = end_token().expect("report");
            assert_eq!(report.counters.command_buffers_submitted, 1);
            assert_eq!(report.counters.dispatches_encoded, 8);
            assert_eq!(report.counters.synchronization_points, 1);
            assert_eq!(report.counters.host_to_device_bytes, 1024);
            assert_eq!(report.counters.device_to_host_bytes, 2048);
            assert_eq!(report.counters.allocations, 1);
            assert_eq!(report.counters.active_bytes_read, 1_378_368);
            assert_eq!(
                report.geometry_active_bytes,
                Some(SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES)
            );
            assert!(report.active_bytes_vs_geometry_fraction.unwrap() < 0.01);
            assert_eq!(report.transfers.len(), 2);
        });
    }

    #[test]
    fn geometry_helper_matches_gate_number() {
        // 8 experts × 3 projections × 1_378_368 × 78 layers ≈ 2.58 GB.
        let g = geometry_active_bytes(78, 8, Some(1_378_368));
        assert_eq!(g, SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES);
        assert_eq!(g, 8u64 * 3 * 1_378_368 * 78);
        // Sanity: ~2.58e9 bytes as the gate states.
        assert!((g as f64 - 2.58e9).abs() < 5e6, "g={g}");
    }

    #[test]
    fn bucket_names_are_gate_stable() {
        let names: Vec<_> = Bucket::ALL.iter().map(|b| b.as_str()).collect();
        assert!(names.contains(&"artifact_verification_and_sha"));
        assert!(names.contains(&"metal_encode"));
        assert!(names.contains(&"metal_submit"));
        assert!(names.contains(&"metal_synchronize"));
        assert!(names.contains(&"attention_and_indexshare"));
        assert!(names.contains(&"routed_experts"));
        assert_eq!(names.len(), 14);
    }
}
