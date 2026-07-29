//! hide-program-runtime: the sandboxed programmatic tool runtime (Bible Book V,
//! sec 18-19).
//!
//! A HIDE agent does not only call one tool at a time - it can run a small
//! *program* that fans out over read handles, filters, ranks, joins, and
//! reduces the results into a structured answer, all inside a sandbox. This
//! crate is that sandbox and its evaluator.
//!
//! # The sandbox in one paragraph
//!
//! A [`Program`] is a tree of [`Expr`] nodes (pure data - it serializes to
//! JSON). It is evaluated by a deterministic in-crate interpreter that has NO
//! ambient authority. The only way a program reaches the outside world is by
//! calling a handle from the closed, read-only [`HandleName`] set, and only if
//! the host granted it: `search.text`, `search.symbol`, `index.references`,
//! `file.read`, `git.diff`, `git.log`, `diagnostic.list`, `test.result.read`,
//! `artifact.read`, `mcp.readonly`. There is no filesystem-write,
//! subprocess, network-egress, environment, or credential handle to name, so a
//! program cannot express one. The clock is virtual and the rng is seeded, so a
//! run is reproducible byte for byte.
//!
//! # Write separation
//!
//! A program may *prepare* a mutation but never commit one. Building a write
//! (an edit, a shell command, a network call, an external mutation) produces a
//! typed [`WriteProposal`] that is collected and returned in [`RunOutput`]; the
//! runtime executes none of them. The proposal travels the normal action plane
//! where real approval + execution live, outside this crate.
//!
//! # Built-in operators
//!
//! `parallel_map`, `bounded_map`, `filter`, `rank`, `group`, `join`, `reduce`,
//! `pagination`, `retry_with_policy`, `schema_validate`, `dedup`, `sample`,
//! `spill_to_artifact`, and `citation_preservation` are provided as [`Operator`]
//! nodes. Iteration exists only through these bounded operators, which keeps
//! evaluation total and every dimension metered.
//!
//! # Enforced limits
//!
//! [`Limits`] bounds instruction count, virtual wall time, peak memory, output
//! bytes, tool calls, map concurrency, per-artifact bytes, and recursion depth.
//! Exhausting any one raises a typed [`RuntimeError::LimitExceeded`] carrying the
//! [`LimitKind`].
//!
//! # Model-free
//!
//! This crate is entirely model-free (RIP doctrine). It evaluates data-shaped
//! programs over host-supplied read handles and proves itself with deterministic
//! tests over fixtures. It never runs a model, opens a socket, spawns a process,
//! or touches the filesystem. Binding a real model to author these programs is a
//! job for a model-bearing layer and is out of scope here; see
//! `DEFERRED_MODEL_REQUIRED` below.
//!
//! DEFERRED_MODEL_REQUIRED: nothing in this crate synthesizes a program from a
//! natural-language goal. Program authoring by a model is deferred; the runtime
//! only *executes* programs it is handed, deterministically.
//!
//! # Example program
//!
//! Fan out a search, project + rank the hits while preserving their citations,
//! and prepare (but do not execute) a follow-up edit.
//!
//! ```
//! use crate::program_runtime::{
//!     run, BinOp, Expr, HandleGrants, HandleName, Lambda, Limits, Operator, Order,
//!     Program, Value, FnHost, map_of, Citation,
//! };
//!
//! // A host that answers `search.text` with two cited rows. Read-only.
//! let host = FnHost::new(|handle, _args, _attempt| {
//!     assert_eq!(handle, HandleName::SearchText);
//!     let row = |path: &str, score: i64| {
//!         map_of([("path", Value::from(path)), ("score", Value::from(score))])
//!             .with_merged_citations(&[Citation::new("search.text").with_locator(path)])
//!     };
//!     Ok(Value::List(vec![row("a.rs", 2), row("b.rs", 9)]))
//! });
//!
//! // Program: search -> keep the path field (preserving citations) -> rank by
//! // score desc -> also stage an edit proposal, and return both.
//! let hits = Expr::handle(HandleName::SearchText, Expr::lit("needle"));
//! let projected = Expr::op(Operator::CitationPreservation {
//!     input: Box::new(hits),
//!     func: Lambda::new(
//!         "h",
//!         Expr::map_lit([("path", Expr::field(Expr::var("h"), ["path"]))]),
//!     ),
//! });
//! let ranked = Expr::op(Operator::Rank {
//!     input: Box::new(Expr::handle(HandleName::SearchText, Expr::lit("needle"))),
//!     key: Lambda::new("h", Expr::field(Expr::var("h"), ["score"])),
//!     order: Order::Desc,
//!     limit: Some(1),
//! });
//! let proposal = Expr::propose_write(Expr::map_lit([
//!     ("kind", Expr::lit("edit")),
//!     ("summary", Expr::lit("rename symbol in top hit")),
//! ]));
//! let root = Expr::map_lit([
//!     ("projected", projected),
//!     ("top", ranked),
//!     ("staged", proposal),
//! ]);
//!
//! let program = Program::new(root);
//! let out = run(&program, &host, &HandleGrants::of([HandleName::SearchText]), Limits::strict())
//!     .expect("program runs");
//!
//! // Citations survived the projection.
//! let projected = out.value.get_path(&["projected".into()]).unwrap();
//! let first = &projected.as_list().unwrap()[0];
//! assert_eq!(first.citations().len(), 1);
//!
//! // The edit was prepared, not executed: one proposal, and it went nowhere.
//! assert_eq!(out.proposals.len(), 1);
//! assert_eq!(out.proposals[0].summary, "rename symbol in top hit");
//! ```

pub use ast::{
    BinOp, Expr, JoinKind, Lambda, Operator, Order, Program, RetryPolicy, SchemaField, SchemaSpec,
};
pub use error::{HandleError, LimitKind, Result, RuntimeError};
pub use handles::{DenyAllHost, FnHost, HandleGrants, HandleName, HostHandles};
pub use interp::{run, Artifact, RunOutput};
pub use limits::{Limits, Usage};
pub use proposal::{WriteKind, WriteProposal};
pub use value::{map_of, Citation, Value, CITATIONS_KEY};

// --- inlined program_runtime/ast.rs ---
pub mod ast {
//! The program AST.
//!
//! A program is a tree of [`Expr`] nodes evaluated by the interpreter in
//! `interp`. The tree is `serde`-serializable, so a program is *data*: it can be
//! authored as JSON, stored, hashed, and replayed. There are no loops or
//! unbounded recursion primitives - iteration happens only through the bounded
//! collection [`Operator`]s, which is what keeps evaluation total and metered.
//!
//! The AST has no node that touches the world. The only outward edge is
//! [`Expr::Handle`], which names a read-only [`HandleName`] and is gated by
//! grants at run time. Mutation is expressed with [`Expr::ProposeWrite`], which
//! builds a proposal and returns - it never executes.

use serde::{Deserialize, Serialize};

use crate::program_runtime::handles::HandleName;
use crate::program_runtime::value::Value;

/// A single-argument function used by collection operators. When applied, `param`
/// is bound to the current element and `body` is evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lambda {
    pub param: String,
    pub body: Box<Expr>,
}

impl Lambda {
    pub fn new(param: impl Into<String>, body: Expr) -> Self {
        Lambda {
            param: param.into(),
            body: Box::new(body),
        }
    }
}

/// A comparison / boolean / arithmetic operator used inside [`Expr::BinOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Add,
    Sub,
    Mul,
    /// String / list containment: `lhs contains rhs`.
    Contains,
}

/// Sort direction for `rank`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    Asc,
    Desc,
}

/// Join flavor for `join`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinKind {
    /// Only rows with a match on both sides.
    Inner,
    /// Every left row; unmatched left rows get a null `right`.
    Left,
}

/// Retry policy for `retry_with_policy`. Backoff is virtual (it advances the
/// runtime clock and counts against the wall-time budget), never a real sleep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    /// Milliseconds of virtual backoff added before attempt N (linear in N).
    pub backoff_ms: u64,
}

impl RetryPolicy {
    pub fn new(max_attempts: u32, backoff_ms: u64) -> Self {
        Self {
            max_attempts,
            backoff_ms,
        }
    }
}

/// A minimal, deterministic schema used by `schema_validate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum SchemaSpec {
    Any,
    Null,
    Bool,
    Int,
    Float,
    /// Any number (int or float).
    Number,
    Str,
    /// A homogeneous list.
    List { items: Box<SchemaSpec> },
    /// An object with typed, optionally-required fields.
    Map { fields: Vec<SchemaField> },
}

/// One field in a [`SchemaSpec::Map`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaField {
    pub name: String,
    pub schema: SchemaSpec,
    #[serde(default)]
    pub required: bool,
}

/// A built-in collection / control operator. Each is deterministic and bounded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum Operator {
    /// Map `func` over a list. `concurrency` is the requested logical width,
    /// checked against the concurrency limit; results are always produced in
    /// input order so output is deterministic regardless of width.
    ParallelMap {
        input: Box<Expr>,
        func: Lambda,
        #[serde(default)]
        concurrency: Option<u32>,
    },
    /// Map `func` over a list with an explicit in-flight `bound`.
    BoundedMap {
        input: Box<Expr>,
        func: Lambda,
        bound: u32,
    },
    /// Keep the elements for which `pred` is truthy.
    Filter { input: Box<Expr>, pred: Lambda },
    /// Sort by `key`, optionally keep the top `limit`.
    Rank {
        input: Box<Expr>,
        key: Lambda,
        order: Order,
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Group elements by `key`; yields a sorted list of `{key, items}` records.
    Group { input: Box<Expr>, key: Lambda },
    /// Join two lists on matching keys; yields `{left, right}` records.
    Join {
        left: Box<Expr>,
        right: Box<Expr>,
        left_key: Lambda,
        right_key: Lambda,
        kind: JoinKind,
    },
    /// Fold a list into a single value. `acc` and `item` are the accumulator and
    /// element variable names bound while evaluating `body`.
    Reduce {
        input: Box<Expr>,
        init: Box<Expr>,
        acc: String,
        item: String,
        body: Box<Expr>,
    },
    /// Take one page of a list.
    Paginate {
        input: Box<Expr>,
        page_size: usize,
        page: usize,
    },
    /// Evaluate `body`, retrying on a retryable handle error per `policy`.
    RetryWithPolicy {
        body: Box<Expr>,
        policy: RetryPolicy,
    },
    /// Validate a value against a schema; returns the value or a schema error.
    SchemaValidate {
        input: Box<Expr>,
        schema: SchemaSpec,
    },
    /// Remove duplicates. With `key`, dedup by the key; without, by whole value.
    /// First occurrence wins; order is preserved.
    Dedup {
        input: Box<Expr>,
        #[serde(default)]
        key: Option<Lambda>,
    },
    /// Deterministically sample up to `k` elements using the seeded rng.
    Sample { input: Box<Expr>, k: usize },
    /// Spill a large value to an artifact and return a reference. Enforces the
    /// per-artifact byte budget.
    SpillToArtifact {
        input: Box<Expr>,
        name: String,
    },
    /// Map `func` over records while preserving each source record's citations
    /// onto the corresponding output record (merged, deduplicated).
    CitationPreservation {
        input: Box<Expr>,
        func: Lambda,
    },
}

/// An expression node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "expr")]
pub enum Expr {
    /// A literal value.
    Lit { value: Value },
    /// A variable reference (a lambda / let binding).
    Var { name: String },
    /// Follow a key path into a value.
    Field { base: Box<Expr>, path: Vec<String> },
    /// Build a list from element expressions.
    List { items: Vec<Expr> },
    /// Build a map from key / value-expression pairs.
    MapLit { entries: Vec<(String, Expr)> },
    /// Call a granted read handle with an argument value.
    Handle { name: HandleName, args: Box<Expr> },
    /// Bind `name` to `value` inside `body`.
    Let {
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    /// Choose a branch by a truthy condition.
    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        otherwise: Box<Expr>,
    },
    /// A binary operation.
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Logical negation of a truthy value.
    Not { value: Box<Expr> },
    /// Prepare (do not execute) a write. The argument is a map describing the
    /// proposal; see `interp` for the accepted shape.
    ProposeWrite { spec: Box<Expr> },
    /// Apply a built-in operator.
    Op { operator: Box<Operator> },
}

// -- ergonomic constructors ---------------------------------------------------

impl Expr {
    pub fn lit(value: impl Into<Value>) -> Expr {
        Expr::Lit {
            value: value.into(),
        }
    }

    pub fn var(name: impl Into<String>) -> Expr {
        Expr::Var { name: name.into() }
    }

    pub fn field(base: Expr, path: impl IntoIterator<Item = &'static str>) -> Expr {
        Expr::Field {
            base: Box::new(base),
            path: path.into_iter().map(str::to_string).collect(),
        }
    }

    pub fn list(items: impl IntoIterator<Item = Expr>) -> Expr {
        Expr::List {
            items: items.into_iter().collect(),
        }
    }

    pub fn map_lit(entries: impl IntoIterator<Item = (&'static str, Expr)>) -> Expr {
        Expr::MapLit {
            entries: entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        }
    }

    pub fn handle(name: HandleName, args: Expr) -> Expr {
        Expr::Handle {
            name,
            args: Box::new(args),
        }
    }

    pub fn let_(name: impl Into<String>, value: Expr, body: Expr) -> Expr {
        Expr::Let {
            name: name.into(),
            value: Box::new(value),
            body: Box::new(body),
        }
    }

    pub fn if_(cond: Expr, then: Expr, otherwise: Expr) -> Expr {
        Expr::If {
            cond: Box::new(cond),
            then: Box::new(then),
            otherwise: Box::new(otherwise),
        }
    }

    pub fn bin(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn not(value: Expr) -> Expr {
        Expr::Not {
            value: Box::new(value),
        }
    }

    pub fn propose_write(spec: Expr) -> Expr {
        Expr::ProposeWrite {
            spec: Box::new(spec),
        }
    }

    pub fn op(operator: Operator) -> Expr {
        Expr::Op {
            operator: Box::new(operator),
        }
    }
}

/// A complete program: the root expression plus the deterministic seed used by
/// `sample` and the clock start. Programs are pure data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub root: Expr,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub clock_start_ms: u64,
}

impl Program {
    pub fn new(root: Expr) -> Self {
        Self {
            root,
            seed: 0,
            clock_start_ms: 0,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}
}


// --- inlined program_runtime/error.rs ---
pub mod error {
//! Typed errors raised by the runtime.
//!
//! Every failure mode is a distinct, matchable variant. The limiter raises
//! [`RuntimeError::LimitExceeded`] with a [`LimitKind`] so a host can tell which
//! budget tripped without string-matching.

use thiserror::Error;

/// The kind of resource budget that was exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    /// Too many AST nodes evaluated.
    Instruction,
    /// The virtual wall clock advanced past the budget.
    WallTime,
    /// A produced value exceeded the peak-memory budget.
    Memory,
    /// The serialized program result exceeded the output budget.
    OutputBytes,
    /// Too many host handles were called.
    ToolCall,
    /// A map operator requested more concurrency than allowed.
    Concurrency,
    /// A spilled artifact exceeded the per-artifact byte budget.
    ArtifactByte,
    /// Evaluation nested deeper than allowed.
    Recursion,
}

impl LimitKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            LimitKind::Instruction => "instruction",
            LimitKind::WallTime => "wall_time",
            LimitKind::Memory => "memory",
            LimitKind::OutputBytes => "output_bytes",
            LimitKind::ToolCall => "tool_call",
            LimitKind::Concurrency => "concurrency",
            LimitKind::ArtifactByte => "artifact_byte",
            LimitKind::Recursion => "recursion",
        }
    }
}

/// Failure surfaced by a host handle. Kept separate from [`RuntimeError`] so a
/// host can report a read failure (missing file, denied scope) without pretending
/// it is a runtime bug. A handle can mark a failure retryable so
/// `retry_with_policy` will re-attempt it.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("handle {handle} failed: {message}")]
pub struct HandleError {
    pub handle: String,
    pub message: String,
    pub retryable: bool,
}

impl HandleError {
    pub fn new(handle: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            handle: handle.into(),
            message: message.into(),
            retryable: false,
        }
    }

    pub fn retryable(handle: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            handle: handle.into(),
            message: message.into(),
            retryable: true,
        }
    }
}

/// Anything that can go wrong while running a program.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RuntimeError {
    /// A resource budget was exhausted. This is the sandbox doing its job, not a
    /// program bug.
    #[error("{kind} limit exceeded (limit {limit}, needed {needed})", kind = kind.as_str())]
    LimitExceeded {
        kind: LimitKind,
        limit: u64,
        needed: u64,
    },

    /// The program called a handle it was not granted. There is no way to
    /// escalate: the host decides grants, the runtime only enforces them.
    #[error("handle {0} is not granted to this program")]
    HandleNotGranted(String),

    /// A referenced variable was not bound in scope.
    #[error("unbound variable: {0}")]
    UnboundVariable(String),

    /// An operation received a value of the wrong shape.
    #[error("type error: {0}")]
    Type(String),

    /// A value failed `schema_validate`.
    #[error("schema validation failed: {0}")]
    Schema(String),

    /// A host handle returned an error and it was not (or no longer) retryable.
    #[error(transparent)]
    Handle(#[from] HandleError),
}

impl RuntimeError {
    pub fn limit(kind: LimitKind, limit: u64, needed: u64) -> Self {
        RuntimeError::LimitExceeded { kind, limit, needed }
    }

    /// The [`LimitKind`] this error carries, if it is a limit error.
    pub fn limit_kind(&self) -> Option<LimitKind> {
        match self {
            RuntimeError::LimitExceeded { kind, .. } => Some(*kind),
            _ => None,
        }
    }
}

/// Convenience result alias for runtime operations.
pub type Result<T> = std::result::Result<T, RuntimeError>;
}


// --- inlined program_runtime/handles.rs ---
pub mod handles {
//! Host handles: the entire external surface of the sandbox.
//!
//! A program cannot touch the world except by calling a handle from the closed
//! [`HandleName`] set below, and only if the host granted it. This is where "no
//! ambient authority" is realized *by construction*: the enum contains only
//! read-oriented handles. There is no filesystem-write, subprocess-spawn,
//! network-egress, environment-read, or credential handle to name, so no program
//! can express one. A write is never a handle - it is a [`WriteProposal`] handed
//! back to the caller (see `proposal`).
//!
//! The handle names mirror the read tool surface a HIDE agent already exposes:
//! `search.text`, `search.symbol`, `index.references`, `file.read`, `git.diff`,
//! `git.log`, `diagnostic.list`, `test.result.read`, `artifact.read`,
//! `mcp.readonly`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::program_runtime::error::HandleError;
use crate::program_runtime::value::Value;

/// The closed set of read-oriented capabilities a program may invoke. This is
/// the complete list; adding a mutating capability would require editing this
/// enum, which is the point - the surface is small, auditable, and read-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HandleName {
    /// Full-text search over the workspace.
    #[serde(rename = "search.text")]
    SearchText,
    /// Symbol / definition search.
    #[serde(rename = "search.symbol")]
    SearchSymbol,
    /// Reference / call-site lookup from the index.
    #[serde(rename = "index.references")]
    IndexReferences,
    /// Read the contents of a file (read-only).
    #[serde(rename = "file.read")]
    FileRead,
    /// Read a diff between two revisions.
    #[serde(rename = "git.diff")]
    GitDiff,
    /// Read commit history.
    #[serde(rename = "git.log")]
    GitLog,
    /// List diagnostics (compiler / linter output).
    #[serde(rename = "diagnostic.list")]
    DiagnosticList,
    /// Read a recorded test result.
    #[serde(rename = "test.result.read")]
    TestResultRead,
    /// Read a stored artifact.
    #[serde(rename = "artifact.read")]
    ArtifactRead,
    /// Call a read-only MCP method (side-effect-free by contract on the host).
    #[serde(rename = "mcp.readonly")]
    McpReadonly,
}

impl HandleName {
    /// Every handle, in a stable order.
    pub const ALL: [HandleName; 10] = [
        HandleName::SearchText,
        HandleName::SearchSymbol,
        HandleName::IndexReferences,
        HandleName::FileRead,
        HandleName::GitDiff,
        HandleName::GitLog,
        HandleName::DiagnosticList,
        HandleName::TestResultRead,
        HandleName::ArtifactRead,
        HandleName::McpReadonly,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            HandleName::SearchText => "search.text",
            HandleName::SearchSymbol => "search.symbol",
            HandleName::IndexReferences => "index.references",
            HandleName::FileRead => "file.read",
            HandleName::GitDiff => "git.diff",
            HandleName::GitLog => "git.log",
            HandleName::DiagnosticList => "diagnostic.list",
            HandleName::TestResultRead => "test.result.read",
            HandleName::ArtifactRead => "artifact.read",
            HandleName::McpReadonly => "mcp.readonly",
        }
    }

    pub fn from_str(s: &str) -> Option<HandleName> {
        HandleName::ALL.into_iter().find(|h| h.as_str() == s)
    }

    /// Documents the invariant: every handle in this enum is read-oriented.
    /// There is intentionally no variant for which this returns false.
    pub const fn is_read_oriented(&self) -> bool {
        true
    }
}

/// The subset of handles a particular program is allowed to call. The runtime
/// checks membership before dispatching; a call to a non-granted handle is a
/// [`crate::program_runtime::error::RuntimeError::HandleNotGranted`], never a silent success.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HandleGrants(BTreeSet<HandleName>);

impl HandleGrants {
    /// Grant nothing. A program with no grants can still compute over its
    /// literals and produce write proposals, but cannot read the world.
    pub fn none() -> Self {
        Self(BTreeSet::new())
    }

    /// Grant every read handle.
    pub fn all() -> Self {
        Self(HandleName::ALL.into_iter().collect())
    }

    /// Grant exactly the listed handles.
    pub fn of<I: IntoIterator<Item = HandleName>>(handles: I) -> Self {
        Self(handles.into_iter().collect())
    }

    pub fn grant(&mut self, handle: HandleName) -> &mut Self {
        self.0.insert(handle);
        self
    }

    pub fn is_granted(&self, handle: HandleName) -> bool {
        self.0.contains(&handle)
    }

    pub fn granted(&self) -> impl Iterator<Item = HandleName> + '_ {
        self.0.iter().copied()
    }
}

/// A host that answers read handles. This is the ONLY trait the runtime calls
/// out through. Implementations must be deterministic and side-effect-free for a
/// given `(handle, args)` if the caller wants byte-identical program output;
/// the runtime does not and cannot enforce that from inside the sandbox, so it
/// is a host contract.
pub trait HostHandles {
    /// Answer one handle call. `attempt` starts at 0 and increments on each
    /// `retry_with_policy` retry, which lets a fixture model a flaky read
    /// deterministically. `args` is the argument value the program passed.
    fn call(&self, handle: HandleName, args: &Value, attempt: u32) -> Result<Value, HandleError>;
}

/// Adapt a closure into a [`HostHandles`]. Handy for tests and doc examples.
pub struct FnHost<F>(F);

impl<F> FnHost<F>
where
    F: Fn(HandleName, &Value, u32) -> Result<Value, HandleError>,
{
    pub fn new(f: F) -> Self {
        FnHost(f)
    }
}

impl<F> HostHandles for FnHost<F>
where
    F: Fn(HandleName, &Value, u32) -> Result<Value, HandleError>,
{
    fn call(&self, handle: HandleName, args: &Value, attempt: u32) -> Result<Value, HandleError> {
        (self.0)(handle, args, attempt)
    }
}

/// A host with no handles at all: every call fails. Useful when a program is
/// expected to be pure (compute over literals, emit proposals) and you want to
/// prove it never reached for the world.
pub struct DenyAllHost;

impl HostHandles for DenyAllHost {
    fn call(&self, handle: HandleName, _args: &Value, _attempt: u32) -> Result<Value, HandleError> {
        Err(HandleError::new(
            handle.as_str(),
            "no host handle is available",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handle_names_roundtrip_dotted_strings() {
        for h in HandleName::ALL {
            assert_eq!(HandleName::from_str(h.as_str()), Some(h));
            let json = serde_json::to_string(&h).unwrap();
            assert_eq!(json, format!("\"{}\"", h.as_str()));
        }
        assert_eq!(HandleName::from_str("fs.write"), None);
        assert_eq!(HandleName::from_str("shell.exec"), None);
    }
    #[test]
    fn grants_are_explicit() {
        let g = HandleGrants::of([HandleName::FileRead]);
        assert!(g.is_granted(HandleName::FileRead));
        assert!(!g.is_granted(HandleName::GitLog));
        assert!(HandleGrants::none().granted().next().is_none());
        assert_eq!(HandleGrants::all().granted().count(), 10);
    }
    #[test]
    fn every_handle_is_read_oriented() {
        assert!(HandleName::ALL.iter().all(|h| h.is_read_oriented()));
    }
}
}


// --- inlined program_runtime/interp.rs ---
pub mod interp {
//! The deterministic tree-walking interpreter.
//!
//! # Why an in-crate interpreter
//!
//! The runtime is a small tree-walking evaluator over the [`Expr`] AST rather
//! than an embedded scripting engine (Lua, JS, wasm, ...). That choice is
//! deliberate: a heavyweight engine brings its own I/O surface, its own clock,
//! its own allocator, and its own nondeterminism, all of which would have to be
//! fenced off again. A purpose-built evaluator has *no* capability we did not
//! give it. There is no `import`, no host-function table beyond the closed
//! [`HandleName`] set, no ambient clock, and no source of entropy except a
//! seeded rng. Determinism and "no ambient authority" fall out of the design
//! instead of being bolted on.
//!
//! Everything here is model-free. The runtime evaluates data-shaped programs
//! over host-provided read handles; it never runs a model.

use crate::program_runtime::ast::{BinOp, Expr, JoinKind, Lambda, Operator, Order, Program, SchemaField, SchemaSpec};
use crate::program_runtime::error::{Result, RuntimeError};
use crate::program_runtime::handles::{HandleGrants, HandleName, HostHandles};
use crate::program_runtime::limits::{Limits, Meter, Usage};
use crate::program_runtime::proposal::{WriteKind, WriteProposal};
use crate::program_runtime::value::{Citation, Value};

/// A spilled artifact produced by `spill_to_artifact`. Held out of the returned
/// value so a large intermediate does not blow the output budget.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Artifact {
    pub id: String,
    pub name: String,
    pub byte_len: u64,
    pub digest: String,
    pub content: Value,
}

/// The result of running a program.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunOutput {
    /// The program's return value.
    pub value: Value,
    /// Mutations the program prepared. NONE were executed by the runtime.
    pub proposals: Vec<WriteProposal>,
    /// Artifacts the program spilled.
    pub artifacts: Vec<Artifact>,
    /// Deterministic resource usage.
    pub usage: Usage,
}

/// A single variable binding in a parent-linked scope chain. Lambda and `let`
/// each introduce exactly one, so the chain is cheap and needs no cloning.
struct Scope<'a> {
    name: &'a str,
    value: &'a Value,
    parent: Option<&'a Scope<'a>>,
}

impl<'a> Scope<'a> {
    fn lookup(&self, name: &str) -> Option<&Value> {
        if self.name == name {
            Some(self.value)
        } else {
            self.parent.and_then(|p| p.lookup(name))
        }
    }
}

fn lookup<'a>(scope: Option<&'a Scope<'a>>, name: &str) -> Option<&'a Value> {
    scope.and_then(|s| s.lookup(name))
}

/// A minimal deterministic rng (SplitMix64). Seeded from the program; no entropy
/// source is consulted, so `sample` is reproducible.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next() % bound as u64) as usize
        }
    }
}

/// Interpreter state for one run.
struct Exec<'h> {
    host: &'h dyn HostHandles,
    grants: &'h HandleGrants,
    meter: Meter,
    rng: SplitMix64,
    proposals: Vec<WriteProposal>,
    artifacts: Vec<Artifact>,
    proposal_seq: u64,
    artifact_seq: u64,
    /// The current attempt index, threaded to handles so a retry can present a
    /// deterministic "next attempt" to a flaky fixture.
    attempt: u32,
}

impl<'h> Exec<'h> {
    fn new(
        host: &'h dyn HostHandles,
        grants: &'h HandleGrants,
        limits: Limits,
        seed: u64,
        clock_start_ms: u64,
    ) -> Self {
        Exec {
            host,
            grants,
            meter: Meter::new(limits, clock_start_ms),
            // Mix the seed so a zero seed is not a degenerate state.
            rng: SplitMix64(seed ^ 0xA5A5_A5A5_5A5A_5A5A),
            proposals: Vec::new(),
            artifacts: Vec::new(),
            proposal_seq: 0,
            artifact_seq: 0,
            attempt: 0,
        }
    }

    fn eval(&mut self, expr: &Expr, scope: Option<&Scope>, depth: u32) -> Result<Value> {
        self.meter.tick_instruction()?;
        self.meter.check_recursion(depth)?;

        match expr {
            Expr::Lit { value } => {
                self.meter.observe_value(value.estimated_bytes())?;
                Ok(value.clone())
            }
            Expr::Var { name } => lookup(scope, name)
                .cloned()
                .ok_or_else(|| RuntimeError::UnboundVariable(name.clone())),
            Expr::Field { base, path } => {
                let b = self.eval(base, scope, depth + 1)?;
                Ok(b.get_path(path).cloned().unwrap_or(Value::Null))
            }
            Expr::List { items } => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(self.eval(it, scope, depth + 1)?);
                }
                let v = Value::List(out);
                self.meter.observe_value(v.estimated_bytes())?;
                Ok(v)
            }
            Expr::MapLit { entries } => {
                let mut m = std::collections::BTreeMap::new();
                for (k, ve) in entries {
                    let val = self.eval(ve, scope, depth + 1)?;
                    m.insert(k.clone(), val);
                }
                let v = Value::Map(m);
                self.meter.observe_value(v.estimated_bytes())?;
                Ok(v)
            }
            Expr::Handle { name, args } => {
                let a = self.eval(args, scope, depth + 1)?;
                self.call_handle(*name, &a)
            }
            Expr::Let { name, value, body } => {
                let bound = self.eval(value, scope, depth + 1)?;
                let inner = Scope {
                    name,
                    value: &bound,
                    parent: scope,
                };
                self.eval(body, Some(&inner), depth + 1)
            }
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                let c = self.eval(cond, scope, depth + 1)?;
                if c.is_truthy() {
                    self.eval(then, scope, depth + 1)
                } else {
                    self.eval(otherwise, scope, depth + 1)
                }
            }
            Expr::BinOp { op, lhs, rhs } => self.eval_binop(*op, lhs, rhs, scope, depth),
            Expr::Not { value } => {
                let v = self.eval(value, scope, depth + 1)?;
                Ok(Value::Bool(!v.is_truthy()))
            }
            Expr::ProposeWrite { spec } => {
                let s = self.eval(spec, scope, depth + 1)?;
                self.propose_write(&s)
            }
            Expr::Op { operator } => self.eval_op(operator, scope, depth),
        }
    }

    fn eval_binop(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        scope: Option<&Scope>,
        depth: u32,
    ) -> Result<Value> {
        // Short-circuit the boolean connectives.
        match op {
            BinOp::And => {
                let l = self.eval(lhs, scope, depth + 1)?;
                if !l.is_truthy() {
                    return Ok(Value::Bool(false));
                }
                let r = self.eval(rhs, scope, depth + 1)?;
                return Ok(Value::Bool(r.is_truthy()));
            }
            BinOp::Or => {
                let l = self.eval(lhs, scope, depth + 1)?;
                if l.is_truthy() {
                    return Ok(Value::Bool(true));
                }
                let r = self.eval(rhs, scope, depth + 1)?;
                return Ok(Value::Bool(r.is_truthy()));
            }
            _ => {}
        }

        let l = self.eval(lhs, scope, depth + 1)?;
        let r = self.eval(rhs, scope, depth + 1)?;
        let out = match op {
            BinOp::Eq => Value::Bool(l == r),
            BinOp::Ne => Value::Bool(l != r),
            BinOp::Lt => Value::Bool(l.total_cmp(&r).is_lt()),
            BinOp::Le => Value::Bool(l.total_cmp(&r).is_le()),
            BinOp::Gt => Value::Bool(l.total_cmp(&r).is_gt()),
            BinOp::Ge => Value::Bool(l.total_cmp(&r).is_ge()),
            BinOp::Add => arith(&l, &r, |a, b| a + b, |a, b| a.wrapping_add(b))?,
            BinOp::Sub => arith(&l, &r, |a, b| a - b, |a, b| a.wrapping_sub(b))?,
            BinOp::Mul => arith(&l, &r, |a, b| a * b, |a, b| a.wrapping_mul(b))?,
            BinOp::Contains => Value::Bool(contains(&l, &r)),
            BinOp::And | BinOp::Or => unreachable!("handled above"),
        };
        Ok(out)
    }

    fn call_handle(&mut self, name: HandleName, args: &Value) -> Result<Value> {
        if !self.grants.is_granted(name) {
            return Err(RuntimeError::HandleNotGranted(name.as_str().to_string()));
        }
        self.meter.charge_tool_call()?;
        let out = self.host.call(name, args, self.attempt)?;
        self.meter.observe_value(out.estimated_bytes())?;
        Ok(out)
    }

    fn propose_write(&mut self, spec: &Value) -> Result<Value> {
        let m = spec
            .as_map()
            .ok_or_else(|| RuntimeError::Type("propose_write expects a map".into()))?;
        let kind_str = m
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| RuntimeError::Type("propose_write: missing 'kind' string".into()))?;
        let kind = WriteKind::from_str(kind_str)
            .ok_or_else(|| RuntimeError::Type(format!("propose_write: unknown kind {kind_str:?}")))?;
        let summary = m
            .get("summary")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let payload = m.get("payload").cloned().unwrap_or(Value::Null);
        let citations = m.get("citations").map(Citation::list_from).unwrap_or_default();

        let id = format!("wp-{}", self.proposal_seq);
        self.proposal_seq += 1;
        self.proposals.push(WriteProposal {
            id: id.clone(),
            kind,
            summary: summary.clone(),
            payload,
            citations,
        });

        // Return a reference the program can embed in its result. The mutation
        // itself is NOT executed here or anywhere in this crate.
        Ok(crate::program_runtime::value::map_of([
            ("@write_proposal", Value::Str(id)),
            ("kind", Value::Str(kind.as_str().to_string())),
            ("summary", Value::Str(summary)),
        ]))
    }

    // -- operators ---------------------------------------------------------

    fn eval_list(&mut self, e: &Expr, scope: Option<&Scope>, depth: u32) -> Result<Vec<Value>> {
        let v = self.eval(e, scope, depth + 1)?;
        match v {
            Value::List(items) => Ok(items),
            other => Err(RuntimeError::Type(format!(
                "expected a list, got {}",
                type_name(&other)
            ))),
        }
    }

    fn apply(
        &mut self,
        lambda: &Lambda,
        arg: &Value,
        scope: Option<&Scope>,
        depth: u32,
    ) -> Result<Value> {
        let inner = Scope {
            name: &lambda.param,
            value: arg,
            parent: scope,
        };
        self.eval(&lambda.body, Some(&inner), depth + 1)
    }

    fn eval_op(&mut self, op: &Operator, scope: Option<&Scope>, depth: u32) -> Result<Value> {
        match op {
            Operator::ParallelMap {
                input,
                func,
                concurrency,
            } => {
                let width = concurrency.unwrap_or(1);
                self.meter.check_concurrency(width)?;
                self.map_over(input, func, scope, depth)
            }
            Operator::BoundedMap { input, func, bound } => {
                self.meter.check_concurrency(*bound)?;
                self.map_over(input, func, scope, depth)
            }
            Operator::Filter { input, pred } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut out = Vec::new();
                for it in &items {
                    if self.apply(pred, it, scope, depth)?.is_truthy() {
                        out.push(it.clone());
                    }
                }
                self.finish_list(out)
            }
            Operator::Rank {
                input,
                key,
                order,
                limit,
            } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut keyed: Vec<(Value, Value)> = Vec::with_capacity(items.len());
                for it in &items {
                    let k = self.apply(key, it, scope, depth)?;
                    keyed.push((k, it.clone()));
                }
                keyed.sort_by(|a, b| {
                    let ord = a.0.total_cmp(&b.0);
                    match order {
                        Order::Asc => ord,
                        Order::Desc => ord.reverse(),
                    }
                });
                let mut out: Vec<Value> = keyed.into_iter().map(|(_, v)| v).collect();
                if let Some(n) = limit {
                    out.truncate(*n);
                }
                self.finish_list(out)
            }
            Operator::Group { input, key } => {
                let items = self.eval_list(input, scope, depth)?;
                // Sorted by canonical key so group order is deterministic.
                let mut groups: std::collections::BTreeMap<String, (Value, Vec<Value>)> =
                    std::collections::BTreeMap::new();
                for it in &items {
                    let k = self.apply(key, it, scope, depth)?;
                    let ck = k.canonical_key();
                    groups
                        .entry(ck)
                        .or_insert_with(|| (k.clone(), Vec::new()))
                        .1
                        .push(it.clone());
                }
                let out: Vec<Value> = groups
                    .into_iter()
                    .map(|(_, (k, items))| {
                        crate::program_runtime::value::map_of([("key", k), ("items", Value::List(items))])
                    })
                    .collect();
                self.finish_list(out)
            }
            Operator::Join {
                left,
                right,
                left_key,
                right_key,
                kind,
            } => self.eval_join(left, right, left_key, right_key, *kind, scope, depth),
            Operator::Reduce {
                input,
                init,
                acc,
                item,
                body,
            } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut acc_val = self.eval(init, scope, depth + 1)?;
                for it in &items {
                    let new = {
                        let s_acc = Scope {
                            name: acc,
                            value: &acc_val,
                            parent: scope,
                        };
                        let s_item = Scope {
                            name: item,
                            value: it,
                            parent: Some(&s_acc),
                        };
                        self.eval(body, Some(&s_item), depth + 1)?
                    };
                    acc_val = new;
                }
                Ok(acc_val)
            }
            Operator::Paginate {
                input,
                page_size,
                page,
            } => {
                let items = self.eval_list(input, scope, depth)?;
                let start = page.saturating_mul(*page_size);
                let out: Vec<Value> = items
                    .into_iter()
                    .skip(start)
                    .take(*page_size)
                    .collect();
                self.finish_list(out)
            }
            Operator::RetryWithPolicy { body, policy } => {
                let attempts = policy.max_attempts.max(1);
                let saved = self.attempt;
                let mut last: Option<RuntimeError> = None;
                for attempt in 0..attempts {
                    if attempt > 0 {
                        self.meter
                            .advance_clock(policy.backoff_ms.saturating_mul(attempt as u64))?;
                    }
                    self.attempt = attempt;
                    match self.eval(body, scope, depth + 1) {
                        Ok(v) => {
                            self.attempt = saved;
                            return Ok(v);
                        }
                        Err(RuntimeError::Handle(h))
                            if h.retryable && attempt + 1 < attempts =>
                        {
                            last = Some(RuntimeError::Handle(h));
                        }
                        Err(e) => {
                            self.attempt = saved;
                            return Err(e);
                        }
                    }
                }
                self.attempt = saved;
                Err(last.unwrap_or_else(|| {
                    RuntimeError::Type("retry_with_policy: no attempts ran".into())
                }))
            }
            Operator::SchemaValidate { input, schema } => {
                let v = self.eval(input, scope, depth + 1)?;
                validate(&v, schema, "$").map_err(RuntimeError::Schema)?;
                Ok(v)
            }
            Operator::Dedup { input, key } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut seen: std::collections::BTreeSet<String> =
                    std::collections::BTreeSet::new();
                let mut out = Vec::new();
                for it in &items {
                    let k = match key {
                        Some(l) => self.apply(l, it, scope, depth)?.canonical_key(),
                        None => it.canonical_key(),
                    };
                    if seen.insert(k) {
                        out.push(it.clone());
                    }
                }
                self.finish_list(out)
            }
            Operator::Sample { input, k } => {
                let items = self.eval_list(input, scope, depth)?;
                let out = self.sample(items, *k);
                self.finish_list(out)
            }
            Operator::SpillToArtifact { input, name } => {
                let v = self.eval(input, scope, depth + 1)?;
                self.spill(v, name)
            }
            Operator::CitationPreservation { input, func } => {
                let items = self.eval_list(input, scope, depth)?;
                let mut out = Vec::with_capacity(items.len());
                for src in &items {
                    let mapped = self.apply(func, src, scope, depth)?;
                    let preserved = mapped.with_merged_citations(&src.citations());
                    out.push(preserved);
                }
                self.finish_list(out)
            }
        }
    }

    fn map_over(
        &mut self,
        input: &Expr,
        func: &Lambda,
        scope: Option<&Scope>,
        depth: u32,
    ) -> Result<Value> {
        let items = self.eval_list(input, scope, depth)?;
        let mut out = Vec::with_capacity(items.len());
        for it in &items {
            out.push(self.apply(func, it, scope, depth)?);
        }
        self.finish_list(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_join(
        &mut self,
        left: &Expr,
        right: &Expr,
        left_key: &Lambda,
        right_key: &Lambda,
        kind: JoinKind,
        scope: Option<&Scope>,
        depth: u32,
    ) -> Result<Value> {
        let lefts = self.eval_list(left, scope, depth)?;
        let rights = self.eval_list(right, scope, depth)?;

        // Index the right side by key, preserving input order within a key.
        let mut index: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, r) in rights.iter().enumerate() {
            let rk = self.apply(right_key, r, scope, depth)?.canonical_key();
            index.entry(rk).or_default().push(i);
        }

        let mut out = Vec::new();
        for l in &lefts {
            let lk = self.apply(left_key, l, scope, depth)?.canonical_key();
            match index.get(&lk) {
                Some(matches) => {
                    for &ri in matches {
                        out.push(join_record(l, &rights[ri]));
                    }
                }
                None => {
                    if matches!(kind, JoinKind::Left) {
                        out.push(join_record(l, &Value::Null));
                    }
                }
            }
        }
        self.finish_list(out)
    }

    fn sample(&mut self, items: Vec<Value>, k: usize) -> Vec<Value> {
        if k >= items.len() {
            return items;
        }
        // Partial Fisher-Yates: select k indices, then return them in original
        // input order for a stable, deterministic subset.
        let mut idx: Vec<usize> = (0..items.len()).collect();
        for i in 0..k {
            let j = i + self.rng.below(items.len() - i);
            idx.swap(i, j);
        }
        let mut chosen: Vec<usize> = idx[..k].to_vec();
        chosen.sort_unstable();
        chosen.into_iter().map(|i| items[i].clone()).collect()
    }

    fn spill(&mut self, v: Value, name: &str) -> Result<Value> {
        let bytes = serde_json::to_vec(&v).map_err(|e| RuntimeError::Type(e.to_string()))?;
        let byte_len = bytes.len() as u64;
        self.meter.check_artifact(byte_len)?;
        let digest = blake3::hash(&bytes).to_hex().to_string();
        let id = format!("artifact-{}", self.artifact_seq);
        self.artifact_seq += 1;
        self.artifacts.push(Artifact {
            id: id.clone(),
            name: name.to_string(),
            byte_len,
            digest: digest.clone(),
            content: v,
        });
        Ok(crate::program_runtime::value::map_of([
            ("@artifact", Value::Str(id)),
            ("name", Value::Str(name.to_string())),
            ("byte_len", Value::Int(byte_len as i64)),
            ("digest", Value::Str(digest)),
        ]))
    }

    fn finish_list(&mut self, out: Vec<Value>) -> Result<Value> {
        let v = Value::List(out);
        self.meter.observe_value(v.estimated_bytes())?;
        Ok(v)
    }
}

fn join_record(l: &Value, r: &Value) -> Value {
    let base = crate::program_runtime::value::map_of([("left", l.clone()), ("right", r.clone())]);
    // Carry both sides' provenance onto the combined record.
    let mut cites = l.citations();
    cites.extend(r.citations());
    base.with_merged_citations(&cites)
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "string",
        Value::List(_) => "list",
        Value::Map(_) => "map",
    }
}

fn arith(
    l: &Value,
    r: &Value,
    ff: impl Fn(f64, f64) -> f64,
    fi: impl Fn(i64, i64) -> i64,
) -> Result<Value> {
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(fi(*a, *b))),
        (a, b) => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => Ok(Value::Float(ff(x, y))),
            _ => Err(RuntimeError::Type(format!(
                "arithmetic on non-numbers: {} and {}",
                type_name(l),
                type_name(r)
            ))),
        },
    }
}

fn contains(haystack: &Value, needle: &Value) -> bool {
    match haystack {
        Value::Str(s) => needle.as_str().map(|n| s.contains(n)).unwrap_or(false),
        Value::List(items) => items.iter().any(|it| it == needle),
        Value::Map(m) => needle
            .as_str()
            .map(|k| m.contains_key(k))
            .unwrap_or(false),
        _ => false,
    }
}

/// Validate a value against a schema. Returns `Ok(())` or a path-qualified
/// message. Deterministic and total.
fn validate(v: &Value, schema: &SchemaSpec, path: &str) -> std::result::Result<(), String> {
    let mismatch = |want: &str| Err(format!("{path}: expected {want}, got {}", type_name(v)));
    match schema {
        SchemaSpec::Any => Ok(()),
        SchemaSpec::Null => matches!(v, Value::Null).then_some(()).ok_or(()).or(mismatch("null")),
        SchemaSpec::Bool => matches!(v, Value::Bool(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("bool")),
        SchemaSpec::Int => matches!(v, Value::Int(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("int")),
        SchemaSpec::Float => matches!(v, Value::Float(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("float")),
        SchemaSpec::Number => matches!(v, Value::Int(_) | Value::Float(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("number")),
        SchemaSpec::Str => matches!(v, Value::Str(_))
            .then_some(())
            .ok_or(())
            .or(mismatch("string")),
        SchemaSpec::List { items } => {
            let Value::List(vs) = v else {
                return mismatch("list");
            };
            for (i, item) in vs.iter().enumerate() {
                validate(item, items, &format!("{path}[{i}]"))?;
            }
            Ok(())
        }
        SchemaSpec::Map { fields } => {
            let Value::Map(m) = v else {
                return mismatch("map");
            };
            for SchemaField {
                name,
                schema,
                required,
            } in fields
            {
                match m.get(name) {
                    Some(fv) => validate(fv, schema, &format!("{path}.{name}"))?,
                    None if *required => {
                        return Err(format!("{path}.{name}: required field missing"));
                    }
                    None => {}
                }
            }
            Ok(())
        }
    }
}

/// Run a program to completion under the given host, grants, and limits.
///
/// The runtime never touches the world except through the granted read
/// [`HandleName`]s, never executes a [`WriteProposal`], and is fully
/// deterministic: the same program, host, grants, and limits produce a
/// byte-identical [`RunOutput`] every time.
pub fn run(
    program: &Program,
    host: &dyn HostHandles,
    grants: &HandleGrants,
    limits: Limits,
) -> Result<RunOutput> {
    let mut exec = Exec::new(host, grants, limits, program.seed, program.clock_start_ms);
    let value = exec.eval(&program.root, None, 0)?;

    // The returned value must fit the output budget.
    let out_bytes = serde_json::to_vec(&value)
        .map(|b| b.len() as u64)
        .unwrap_or(0);
    exec.meter.check_output(out_bytes)?;

    Ok(RunOutput {
        value,
        proposals: exec.proposals,
        artifacts: exec.artifacts,
        usage: exec.meter.usage(),
    })
}

/// The reserved record field name that carries provenance, re-exported for
/// callers building fixtures.
pub use crate::program_runtime::value::CITATIONS_KEY as CITATIONS_FIELD;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_runtime::ast::*;
    use crate::program_runtime::handles::{DenyAllHost, FnHost};
    use crate::program_runtime::value::map_of;
    fn run_pure(root: Expr, limits: Limits) -> Result<RunOutput> {
        let prog = Program::new(root);
        run(&prog, &DenyAllHost, &HandleGrants::none(), limits)
    }
    #[test]
    fn literals_and_arithmetic() {
        let e = Expr::bin(BinOp::Add, Expr::lit(2i64), Expr::lit(3i64));
        let out = run_pure(e, Limits::unbounded()).unwrap();
        assert_eq!(out.value, Value::Int(5));
    }
    #[test]
    fn filter_rank_over_list() {
        let rows = Expr::lit(Value::List(vec![
            map_of([("n", Value::Int(3))]),
            map_of([("n", Value::Int(1))]),
            map_of([("n", Value::Int(2))]),
        ]));
        let prog = Expr::op(Operator::Rank {
            input: Box::new(Expr::op(Operator::Filter {
                input: Box::new(rows),
                pred: Lambda::new(
                    "r",
                    Expr::bin(
                        BinOp::Ge,
                        Expr::field(Expr::var("r"), ["n"]),
                        Expr::lit(2i64),
                    ),
                ),
            })),
            key: Lambda::new("r", Expr::field(Expr::var("r"), ["n"])),
            order: Order::Desc,
            limit: None,
        });
        let out = run_pure(prog, Limits::unbounded()).unwrap();
        let list = out.value.as_list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].get_path(&["n".into()]), Some(&Value::Int(3)));
        assert_eq!(list[1].get_path(&["n".into()]), Some(&Value::Int(2)));
    }
    #[test]
    fn ungranted_handle_is_denied() {
        let e = Expr::handle(HandleName::FileRead, Expr::lit("x"));
        let err = run_pure(e, Limits::unbounded()).unwrap_err();
        assert_eq!(err, RuntimeError::HandleNotGranted("file.read".into()));
    }
    #[test]
    fn granted_handle_is_called() {
        let host = FnHost::new(|h, _a, _t| {
            assert_eq!(h, HandleName::FileRead);
            Ok(Value::Str("hello".into()))
        });
        let prog = Program::new(Expr::handle(HandleName::FileRead, Expr::lit("p")));
        let out = run(
            &prog,
            &host,
            &HandleGrants::of([HandleName::FileRead]),
            Limits::unbounded(),
        )
        .unwrap();
        assert_eq!(out.value, Value::Str("hello".into()));
        assert_eq!(out.usage.tool_calls, 1);
    }
    #[test]
    fn reduce_sums() {
        let rows = Expr::lit(Value::List(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(4),
        ]));
        let prog = Expr::op(Operator::Reduce {
            input: Box::new(rows),
            init: Box::new(Expr::lit(0i64)),
            acc: "a".into(),
            item: "x".into(),
            body: Box::new(Expr::bin(BinOp::Add, Expr::var("a"), Expr::var("x"))),
        });
        let out = run_pure(prog, Limits::unbounded()).unwrap();
        assert_eq!(out.value, Value::Int(7));
    }
    #[test]
    fn dedup_and_paginate() {
        let rows = Expr::lit(Value::List(vec![
            Value::Int(1),
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(3),
        ]));
        let dedup = Expr::op(Operator::Dedup {
            input: Box::new(rows),
            key: None,
        });
        let page = Expr::op(Operator::Paginate {
            input: Box::new(dedup),
            page_size: 2,
            page: 1,
        });
        let out = run_pure(page, Limits::unbounded()).unwrap();
        assert_eq!(out.value, Value::List(vec![Value::Int(3)]));
    }
    #[test]
    fn schema_validate_rejects_bad_shape() {
        let bad = Expr::lit(map_of([("n", Value::Str("x".into()))]));
        let prog = Expr::op(Operator::SchemaValidate {
            input: Box::new(bad),
            schema: SchemaSpec::Map {
                fields: vec![SchemaField {
                    name: "n".into(),
                    schema: SchemaSpec::Int,
                    required: true,
                }],
            },
        });
        let err = run_pure(prog, Limits::unbounded()).unwrap_err();
        assert!(matches!(err, RuntimeError::Schema(_)));
    }
    #[test]
    fn sample_is_seed_deterministic() {
        let rows: Vec<Value> = (0..20).map(Value::Int).collect();
        let mk = Expr::op(Operator::Sample {
            input: Box::new(Expr::lit(Value::List(rows))),
            k: 5,
        });
        let prog = Program::new(mk).with_seed(42);
        let a = run(&prog, &DenyAllHost, &HandleGrants::none(), Limits::unbounded()).unwrap();
        let b = run(&prog, &DenyAllHost, &HandleGrants::none(), Limits::unbounded()).unwrap();
        assert_eq!(a.value, b.value);
        assert_eq!(a.value.as_list().unwrap().len(), 5);
    }
    #[test]
    fn retry_recovers_from_flaky_handle() {
        let host = FnHost::new(|_h, _a, attempt| {
            if attempt < 2 {
                Err(crate::program_runtime::error::HandleError::retryable("git.log", "transient"))
            } else {
                Ok(Value::Str("ok".into()))
            }
        });
        let prog = Program::new(Expr::op(Operator::RetryWithPolicy {
            body: Box::new(Expr::handle(HandleName::GitLog, Expr::lit(Value::Null))),
            policy: RetryPolicy::new(3, 1),
        }));
        let out = run(
            &prog,
            &host,
            &HandleGrants::of([HandleName::GitLog]),
            Limits::unbounded(),
        )
        .unwrap();
        assert_eq!(out.value, Value::Str("ok".into()));
        assert_eq!(out.usage.tool_calls, 3);
    }
}
}


// --- inlined program_runtime/limits.rs ---
pub mod limits {
//! Resource limits and the meter that enforces them.
//!
//! Every dimension the sandbox bounds is a field on [`Limits`]. The [`Meter`]
//! holds the running counters and the virtual clock; each is checked at the
//! point it advances, so exhaustion is caught the instant it happens and turns
//! into a typed [`RuntimeError::LimitExceeded`]. Because time is virtual (there
//! is no real sleep or wall-clock read), a run is deterministic and the
//! wall-time budget is reproducible.

use crate::program_runtime::error::{LimitKind, Result, RuntimeError};

/// The bounds a program runs under. Construct with [`Limits::unbounded`] and
/// tighten, or with [`Limits::strict`] for a conservative default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Maximum AST nodes evaluated.
    pub instructions: u64,
    /// Virtual wall-clock budget, in milliseconds.
    pub wall_time_ms: u64,
    /// Peak single-value memory footprint, in bytes.
    pub memory_bytes: u64,
    /// Maximum serialized size of the program result, in bytes.
    pub output_bytes: u64,
    /// Maximum number of host handle calls.
    pub tool_calls: u32,
    /// Maximum concurrency a map operator may request.
    pub concurrency: u32,
    /// Maximum size of a single spilled artifact, in bytes.
    pub artifact_bytes: u64,
    /// Maximum evaluation nesting depth.
    pub recursion_depth: u32,
    /// Virtual milliseconds charged per handle call. Lets the wall-time budget
    /// be tripped by I/O-shaped work independently of instruction count.
    pub handle_latency_ms: u64,
}

impl Limits {
    /// Effectively no limits. Handy as a base to tighten one dimension at a time.
    pub fn unbounded() -> Self {
        Self {
            instructions: u64::MAX,
            wall_time_ms: u64::MAX,
            memory_bytes: u64::MAX,
            output_bytes: u64::MAX,
            tool_calls: u32::MAX,
            concurrency: u32::MAX,
            artifact_bytes: u64::MAX,
            recursion_depth: u32::MAX,
            handle_latency_ms: 1,
        }
    }

    /// A conservative default suited to a small analysis program.
    pub fn strict() -> Self {
        Self {
            instructions: 100_000,
            wall_time_ms: 5_000,
            memory_bytes: 8 * 1024 * 1024,
            output_bytes: 256 * 1024,
            tool_calls: 128,
            concurrency: 8,
            artifact_bytes: 1024 * 1024,
            recursion_depth: 256,
            handle_latency_ms: 5,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Limits::strict()
    }
}

/// Running counters plus the virtual clock. One meter lives per run.
#[derive(Debug)]
pub struct Meter {
    limits: Limits,
    instructions: u64,
    clock_ms: u64,
    tool_calls: u32,
    peak_memory: u64,
}

impl Meter {
    pub fn new(limits: Limits, clock_start_ms: u64) -> Self {
        Self {
            limits,
            instructions: 0,
            clock_ms: clock_start_ms,
            tool_calls: 0,
            peak_memory: 0,
        }
    }

    /// Charge one evaluated AST node. Called at the top of every eval step.
    pub fn tick_instruction(&mut self) -> Result<()> {
        self.instructions += 1;
        if self.instructions > self.limits.instructions {
            return Err(RuntimeError::limit(
                LimitKind::Instruction,
                self.limits.instructions,
                self.instructions,
            ));
        }
        Ok(())
    }

    /// Advance the virtual clock and check the wall-time budget. Used for handle
    /// latency and retry backoff.
    pub fn advance_clock(&mut self, ms: u64) -> Result<()> {
        self.clock_ms = self.clock_ms.saturating_add(ms);
        if self.clock_ms > self.limits.wall_time_ms {
            return Err(RuntimeError::limit(
                LimitKind::WallTime,
                self.limits.wall_time_ms,
                self.clock_ms,
            ));
        }
        Ok(())
    }

    /// Charge one host handle call (and its latency).
    pub fn charge_tool_call(&mut self) -> Result<()> {
        self.tool_calls += 1;
        if self.tool_calls > self.limits.tool_calls {
            return Err(RuntimeError::limit(
                LimitKind::ToolCall,
                self.limits.tool_calls as u64,
                self.tool_calls as u64,
            ));
        }
        let latency = self.limits.handle_latency_ms;
        self.advance_clock(latency)
    }

    /// Record that a value of `bytes` was produced and check the peak-memory
    /// budget.
    pub fn observe_value(&mut self, bytes: u64) -> Result<()> {
        if bytes > self.peak_memory {
            self.peak_memory = bytes;
        }
        if self.peak_memory > self.limits.memory_bytes {
            return Err(RuntimeError::limit(
                LimitKind::Memory,
                self.limits.memory_bytes,
                self.peak_memory,
            ));
        }
        Ok(())
    }

    /// Check that a requested map concurrency is within budget.
    pub fn check_concurrency(&self, requested: u32) -> Result<()> {
        if requested > self.limits.concurrency {
            return Err(RuntimeError::limit(
                LimitKind::Concurrency,
                self.limits.concurrency as u64,
                requested as u64,
            ));
        }
        Ok(())
    }

    /// Check a recursion depth against the budget.
    pub fn check_recursion(&self, depth: u32) -> Result<()> {
        if depth > self.limits.recursion_depth {
            return Err(RuntimeError::limit(
                LimitKind::Recursion,
                self.limits.recursion_depth as u64,
                depth as u64,
            ));
        }
        Ok(())
    }

    /// Check a serialized output size against the budget.
    pub fn check_output(&self, bytes: u64) -> Result<()> {
        if bytes > self.limits.output_bytes {
            return Err(RuntimeError::limit(
                LimitKind::OutputBytes,
                self.limits.output_bytes,
                bytes,
            ));
        }
        Ok(())
    }

    /// Check a spilled-artifact size against the budget.
    pub fn check_artifact(&self, bytes: u64) -> Result<()> {
        if bytes > self.limits.artifact_bytes {
            return Err(RuntimeError::limit(
                LimitKind::ArtifactByte,
                self.limits.artifact_bytes,
                bytes,
            ));
        }
        Ok(())
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// A deterministic snapshot of consumption, included in the run output.
    pub fn usage(&self) -> Usage {
        Usage {
            instructions: self.instructions,
            clock_ms: self.clock_ms,
            tool_calls: self.tool_calls,
            peak_memory_bytes: self.peak_memory,
        }
    }
}

/// What a run consumed. Deterministic for a given program + host, so it does not
/// perturb byte-identical output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    pub instructions: u64,
    pub clock_ms: u64,
    pub tool_calls: u32,
    pub peak_memory_bytes: u64,
}
}


// --- inlined program_runtime/proposal.rs ---
pub mod proposal {
//! Write separation: a program may *prepare* a mutation but never commit it.
//!
//! The sandbox has no write capability of any kind. When a program wants to edit
//! a file, run a shell command, reach the network, or mutate any external
//! system, it builds a [`WriteProposal`] describing the intended change. The
//! runtime collects these and returns them in the run output; it never executes
//! one. The proposal then travels the normal action plane, where the real
//! approval + execution machinery lives (outside this crate).
//!
//! This keeps the dangerous half of "tool use" out of the deterministic
//! evaluator entirely: nothing the interpreter does can touch the world.

use serde::{Deserialize, Serialize};

use crate::program_runtime::value::{Citation, Value};

/// The category of a prepared mutation. Mirrors the effect classes the action
/// plane knows how to gate and execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteKind {
    /// A file edit (create / modify / delete). `payload` carries path + diff.
    Edit,
    /// A shell command. `payload` carries the command and cwd.
    Shell,
    /// A network request. `payload` carries method + url + body.
    Network,
    /// Any other external mutation (a connector call, a service action).
    ExternalMutation,
}

impl WriteKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            WriteKind::Edit => "edit",
            WriteKind::Shell => "shell",
            WriteKind::Network => "network",
            WriteKind::ExternalMutation => "external_mutation",
        }
    }

    pub fn from_str(s: &str) -> Option<WriteKind> {
        match s {
            "edit" => Some(WriteKind::Edit),
            "shell" => Some(WriteKind::Shell),
            "network" => Some(WriteKind::Network),
            "external_mutation" => Some(WriteKind::ExternalMutation),
            _ => None,
        }
    }
}

/// A prepared, un-executed mutation. Produced by a program, returned to the
/// caller, executed by nobody inside this crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WriteProposal {
    /// A deterministic id assigned by the runtime in creation order
    /// (`wp-0`, `wp-1`, ...). Lets the program reference the proposal in its
    /// result.
    pub id: String,
    pub kind: WriteKind,
    /// A one-line description of the intended change.
    pub summary: String,
    /// The typed detail the action plane needs to execute it (path, diff,
    /// command, url, ...). Opaque to the runtime.
    pub payload: Value,
    /// The evidence the program used to justify this change, carried forward so
    /// the reviewer sees the provenance.
    #[serde(default)]
    pub citations: Vec<Citation>,
}
}


// --- inlined program_runtime/value.rs ---
pub mod value {
//! The deterministic data model that flows through a program.
//!
//! [`Value`] is a JSON-shaped value with a fixed, deterministic serialization:
//! objects are backed by a [`BTreeMap`] so keys are always emitted in sorted
//! order, which is what makes byte-identical output across runs possible. There
//! is deliberately no handle to the outside world in this type: it is pure data.
//!
//! Provenance rides along inside records. A record (a [`Value::Map`]) may carry
//! a reserved [`CITATIONS_KEY`] field holding a list of [`Citation`]s. The
//! citation-preserving operators read and merge that field so that evidence is
//! never silently dropped as data is transformed.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};

/// The reserved record field that carries provenance. A [`Value::Map`] with
/// this key holds a [`Value::List`] of citation objects.
pub const CITATIONS_KEY: &str = "@citations";

/// A piece of evidence that a record was derived from. Structured so it can be
/// re-verified later against a content-addressed store (the `digest`).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Citation {
    /// The source the evidence came from: a handle name, a path, a uri, a
    /// commit, etc. Opaque to the runtime.
    pub source: String,
    /// Where inside the source: a line range, a symbol, a byte span. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// A content digest of the cited bytes, for later re-verification. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    /// A short human-readable excerpt. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

impl Citation {
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            locator: None,
            digest: None,
            snippet: None,
        }
    }

    pub fn with_locator(mut self, locator: impl Into<String>) -> Self {
        self.locator = Some(locator.into());
        self
    }

    pub fn with_digest(mut self, digest: impl Into<String>) -> Self {
        self.digest = Some(digest.into());
        self
    }

    /// A stable key used to dedup citations during merge. Two citations with the
    /// same key are treated as the same piece of evidence.
    pub fn dedup_key(&self) -> String {
        format!(
            "{}|{}|{}",
            self.source,
            self.locator.as_deref().unwrap_or(""),
            self.digest.as_deref().unwrap_or(""),
        )
    }

    fn to_value(&self) -> Value {
        let mut m = BTreeMap::new();
        m.insert("source".to_string(), Value::Str(self.source.clone()));
        if let Some(l) = &self.locator {
            m.insert("locator".to_string(), Value::Str(l.clone()));
        }
        if let Some(d) = &self.digest {
            m.insert("digest".to_string(), Value::Str(d.clone()));
        }
        if let Some(s) = &self.snippet {
            m.insert("snippet".to_string(), Value::Str(s.clone()));
        }
        Value::Map(m)
    }

    /// Parse one citation from a `{source, locator?, digest?, snippet?}` map.
    /// Returns `None` if the value is not a map or lacks a `source` string.
    pub fn from_value(v: &Value) -> Option<Citation> {
        let m = v.as_map()?;
        let source = m.get("source").and_then(Value::as_str)?.to_string();
        Some(Citation {
            source,
            locator: m.get("locator").and_then(Value::as_str).map(str::to_string),
            digest: m.get("digest").and_then(Value::as_str).map(str::to_string),
            snippet: m.get("snippet").and_then(Value::as_str).map(str::to_string),
        })
    }

    /// Parse a [`Value::List`] of citation maps, skipping any that do not parse.
    pub fn list_from(v: &Value) -> Vec<Citation> {
        match v.as_list() {
            Some(items) => items.iter().filter_map(Citation::from_value).collect(),
            None => Vec::new(),
        }
    }
}

/// A JSON-shaped value. Maps are sorted so serialization is canonical.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Value {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Truthiness for predicates: false / null / 0 / empty string / empty
    /// collection are falsey; everything else is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(v) => !v.is_empty(),
            Value::Map(m) => !m.is_empty(),
        }
    }

    /// Follow a dotted path of keys into nested maps. Returns `None` if any
    /// segment is missing or a non-map is traversed.
    pub fn get_path(&self, path: &[String]) -> Option<&Value> {
        let mut cur = self;
        for key in path {
            cur = cur.as_map()?.get(key)?;
        }
        Some(cur)
    }

    /// A deterministic canonical string key for grouping / joining / dedup. It
    /// is the canonical JSON encoding, so it is type-sensitive (integer `1` and
    /// string `"1"` do not collide).
    pub fn canonical_key(&self) -> String {
        serde_json::to_string(self).expect("Value always serializes")
    }

    /// A cheap estimate of the memory footprint of this value, in bytes. Used by
    /// the memory limiter; it does not have to be exact, only monotone and
    /// deterministic.
    pub fn estimated_bytes(&self) -> u64 {
        match self {
            Value::Null | Value::Bool(_) => 1,
            Value::Int(_) | Value::Float(_) => 8,
            Value::Str(s) => s.len() as u64,
            Value::List(v) => 8 + v.iter().map(Value::estimated_bytes).sum::<u64>(),
            Value::Map(m) => {
                8 + m
                    .iter()
                    .map(|(k, v)| k.len() as u64 + v.estimated_bytes())
                    .sum::<u64>()
            }
        }
    }

    fn type_rank(&self) -> u8 {
        match self {
            Value::Null => 0,
            Value::Bool(_) => 1,
            Value::Int(_) | Value::Float(_) => 2,
            Value::Str(_) => 3,
            Value::List(_) => 4,
            Value::Map(_) => 5,
        }
    }

    /// A deterministic total order across all value shapes. Numbers compare
    /// numerically (integers and floats interleave); unlike types order by a
    /// fixed type rank. Used by `rank`.
    pub fn total_cmp(&self, other: &Value) -> Ordering {
        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (a, b) if a.type_rank() == 2 && b.type_rank() == 2 => {
                let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
                x.partial_cmp(&y).unwrap_or(Ordering::Equal)
            }
            (Value::Str(a), Value::Str(b)) => a.cmp(b),
            (Value::List(a), Value::List(b)) => {
                for (x, y) in a.iter().zip(b.iter()) {
                    match x.total_cmp(y) {
                        Ordering::Equal => continue,
                        non_eq => return non_eq,
                    }
                }
                a.len().cmp(&b.len())
            }
            (Value::Map(a), Value::Map(b)) => {
                // Compare by canonical key encoding for a stable, total order.
                a.iter()
                    .map(|(k, v)| (k, v.canonical_key()))
                    .collect::<Vec<_>>()
                    .cmp(&b.iter().map(|(k, v)| (k, v.canonical_key())).collect::<Vec<_>>())
            }
            (a, b) => a.type_rank().cmp(&b.type_rank()),
        }
    }

    // --- citation helpers -------------------------------------------------

    /// Read the citations attached to a record. Returns an empty vector for a
    /// non-map or a record without the reserved field.
    pub fn citations(&self) -> Vec<Citation> {
        let Some(m) = self.as_map() else {
            return Vec::new();
        };
        let Some(Value::List(items)) = m.get(CITATIONS_KEY) else {
            return Vec::new();
        };
        items.iter().filter_map(Citation::from_value).collect()
    }

    /// Return a copy of this record with `extra` citations merged into its
    /// reserved field (deduplicated, stable order). A no-op on a non-map value.
    pub fn with_merged_citations(&self, extra: &[Citation]) -> Value {
        let Value::Map(m) = self else {
            return self.clone();
        };
        let mut merged = self.citations();
        let mut seen: std::collections::BTreeSet<String> =
            merged.iter().map(Citation::dedup_key).collect();
        for c in extra {
            if seen.insert(c.dedup_key()) {
                merged.push(c.clone());
            }
        }
        let mut out = m.clone();
        out.insert(
            CITATIONS_KEY.to_string(),
            Value::List(merged.iter().map(Citation::to_value).collect()),
        );
        Value::Map(out)
    }
}

// -- convenience conversions --------------------------------------------------

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}
impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Int(v as i64)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}
impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::List(v)
    }
}
impl From<BTreeMap<String, Value>> for Value {
    fn from(v: BTreeMap<String, Value>) -> Self {
        Value::Map(v)
    }
}

/// Build a [`Value::Map`] from an iterator of `(key, value)` pairs.
pub fn map_of<I, K, V>(pairs: I) -> Value
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<Value>,
{
    Value::Map(pairs.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
}

// -- serde: emit plain JSON via a serde_json::Value bridge --------------------

impl Value {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Int(i) => serde_json::Value::Number((*i).into()),
            Value::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::Str(s) => serde_json::Value::String(s.clone()),
            Value::List(v) => serde_json::Value::Array(v.iter().map(Value::to_json).collect()),
            Value::Map(m) => {
                // Insert in sorted (BTreeMap) order so the encoding is canonical
                // regardless of whether serde_json preserves insertion order.
                let mut obj = serde_json::Map::new();
                for (k, v) in m {
                    obj.insert(k.clone(), v.to_json());
                }
                serde_json::Value::Object(obj)
            }
        }
    }

    fn from_json(j: serde_json::Value) -> Value {
        match j {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(u) = n.as_u64() {
                    Value::Int(u as i64)
                } else {
                    Value::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            serde_json::Value::String(s) => Value::Str(s),
            serde_json::Value::Array(a) => {
                Value::List(a.into_iter().map(Value::from_json).collect())
            }
            serde_json::Value::Object(o) => {
                Value::Map(o.into_iter().map(|(k, v)| (k, Value::from_json(v))).collect())
            }
        }
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_json().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let j = serde_json::Value::deserialize(deserializer)?;
        Ok(Value::from_json(j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serializes_as_plain_json_with_sorted_keys() {
        let v = map_of([("b", Value::Int(2)), ("a", Value::Int(1))]);
        assert_eq!(serde_json::to_string(&v).unwrap(), r#"{"a":1,"b":2}"#);
    }
    #[test]
    fn roundtrips_through_json() {
        let v = map_of([
            ("n", Value::Null),
            ("f", Value::Float(1.5)),
            ("l", Value::List(vec![Value::Int(1), Value::Str("x".into())])),
        ]);
        let s = serde_json::to_string(&v).unwrap();
        let back: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v, back);
    }
    #[test]
    fn total_cmp_interleaves_int_and_float() {
        assert_eq!(Value::Int(1).total_cmp(&Value::Float(1.5)), Ordering::Less);
        assert_eq!(Value::Float(2.0).total_cmp(&Value::Int(2)), Ordering::Equal);
    }
    #[test]
    fn citations_roundtrip_and_merge_dedup() {
        let rec = map_of([("id", Value::Int(1))]);
        let c1 = Citation::new("file.read").with_locator("L1-L3");
        let merged = rec.with_merged_citations(&[c1.clone(), c1.clone()]);
        assert_eq!(merged.citations(), vec![c1.clone()]);
        let again = merged.with_merged_citations(&[c1.clone()]);
        assert_eq!(again.citations(), vec![c1]);
    }
}
}

