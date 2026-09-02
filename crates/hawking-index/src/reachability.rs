//! Capability-reachability facts, extracted from the existing tree-sitter
//! Python grammar and merkle-diffed incrementally.
//!
//! This is NOT a second AST scanner. It walks the same `tree_sitter_python`
//! CST `parse::parse_source` uses, and emits the import / bound-name / Call /
//! subprocess / tool-dispatch facts that
//! `tools/future/capability_reachability.py` currently derives with CPython's
//! `ast` module. The Python assembler keeps its verdict rules; this crate
//! supplies the facts so it does not have to walk the tree itself.
//!
//! Call-site semantics match the Python engine exactly:
//! - a module import is not a function call
//! - `name()` and `mod.name()` (Name / Attribute-of-Name) count; `a.b.c()` does not
//! - a subprocess path string only counts inside `run`/`Popen`/`check_call`/
//!   `check_output`/`call`
//! - a tool name only counts in `"tool": "…"` / `tool="…"` / `invoke("…")` shape

use crate::merkle::{Blake3MerkleScanner, MerkleKind, MerkleNode, MerkleScanner};
use hide_core::{HideError, Result};
use regex::Regex;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
use tree_sitter::{Node, Parser};

pub const FACTS_SCHEMA: &str = "hawking.index.reachability_facts.v1";
const CACHE_SCHEMA_VERSION: &str = "1";

const SUBPROCESS_NAMES: &[&str] = &["run", "Popen", "check_call", "check_output", "call"];

/// Per-file facts. Cached under (rel_path, content_hash).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileFacts {
    pub imports: Vec<ImportFact>,
    pub binds: Vec<(String, String)>,
    pub calls: Vec<CallFact>,
    pub subprocess: Vec<SubprocessFact>,
    pub literals: Vec<LiteralFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImportFact {
    pub target: String,
    pub line: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CallFact {
    pub line: u32,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qualifier: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubprocessFact {
    pub line: u32,
    pub strings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiteralFact {
    pub line: u32,
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Site {
    pub file: String,
    pub line: u32,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CallSiteDump {
    pub file: String,
    pub line: u32,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qualifier: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubprocessDump {
    pub file: String,
    pub line: u32,
    pub strings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiteralDump {
    pub file: String,
    pub line: u32,
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct IndexStats {
    pub parsed: usize,
    pub reused: usize,
    pub files: usize,
    pub merkle_dirty: usize,
    pub elapsed_ms: u64,
    pub cold: bool,
}

/// JSON document consumed by `capability_reachability.assemble()`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReachabilityDump {
    pub schema: String,
    pub files: Vec<String>,
    pub import_sites: BTreeMap<String, Vec<Site>>,
    /// rel_path → [(local_name, fully_dotted_symbol_or_module), ...]
    pub bound_names: BTreeMap<String, Vec<(String, String)>>,
    pub calls: Vec<CallSiteDump>,
    pub subprocess: Vec<SubprocessDump>,
    pub literals: Vec<LiteralDump>,
    pub index: IndexStats,
}

#[derive(Debug, Clone)]
pub struct CollectOptions {
    pub root: PathBuf,
    pub cache_dir: PathBuf,
}

impl CollectOptions {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let cache_dir = root.join(".hide").join("reachability-index");
        Self { root, cache_dir }
    }

    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }
}

struct ExtractCtx<'a> {
    rel_path: &'a str,
    known_files: &'a HashSet<String>,
}

/// Extract reachability facts from one Python source buffer.
///
/// `known_files` is the repo-relative set of `*.py` paths (posix), used for the
/// sibling-import idiom (`sys.path.insert(dirname(__file__)); from _common import x`).
pub fn extract_python_facts(
    rel_path: &str,
    source: &str,
    known_files: &HashSet<String>,
) -> FileFacts {
    let mut facts = FileFacts::default();
    scan_literals(source, &mut facts);

    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .is_err()
    {
        return facts;
    }
    let Some(tree) = parser.parse(source, None) else {
        return facts;
    };
    let ctx = ExtractCtx {
        rel_path,
        known_files,
    };
    walk(tree.root_node(), source.as_bytes(), &mut facts, &ctx);
    facts
}

fn scan_literals(source: &str, facts: &mut FileFacts) {
    let re = literal_re();
    let mut seen: HashSet<(u32, String)> = HashSet::new();
    for (idx, line) in source.split('\n').enumerate() {
        let line_no = (idx + 1) as u32;
        // Match Python's `str.splitlines()` on `\n`; a trailing empty line after
        // a final newline is *not* emitted by splitlines, but a mid-file empty
        // line is. enumerate over split('\n') can produce a trailing empty
        // extra line — it has no matches, so it is harmless.
        for cap in re.captures_iter(line) {
            let token = cap
                .get(1)
                .or_else(|| cap.get(2))
                .or_else(|| cap.get(3))
                .or_else(|| cap.get(4))
                .map(|m| m.as_str().to_string());
            let Some(token) = token else {
                continue;
            };
            if seen.insert((line_no, token.clone())) {
                facts.literals.push(LiteralFact {
                    line: line_no,
                    token,
                });
            }
        }
    }
}

fn literal_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // The `regex` crate has no backreferences. Spell each quote flavour
        // out so `"tool": "name"` / `tool='name'` / `invoke("name"` match the
        // same way capability_reachability.py's `_tool_dispatch_pattern` does.
        Regex::new(
            r#"(?:"tool"|'tool'|\btool)\s*[:=]\s*"([^"]+)"|(?:"tool"|'tool'|\btool)\s*[:=]\s*'([^']+)'|invoke\(\s*"([^"]+)"|invoke\(\s*'([^']+)'"#,
        )
        .expect("literal dispatch regex")
    })
}

fn walk<'a>(node: Node<'a>, src: &[u8], facts: &mut FileFacts, ctx: &ExtractCtx<'_>) {
    match node.kind() {
        "import_statement" => extract_import(node, src, facts, ctx),
        "import_from_statement" => extract_import_from(node, src, facts, ctx),
        "call" => extract_call(node, src, facts),
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk(child, src, facts, ctx);
    }
}

fn node_line(node: Node<'_>) -> u32 {
    node.start_position().row as u32 + 1
}

fn node_text<'a>(node: Node<'a>, src: &'a [u8]) -> String {
    node.utf8_text(src).unwrap_or("").to_string()
}

fn extract_import(node: Node<'_>, src: &[u8], facts: &mut FileFacts, ctx: &ExtractCtx<'_>) {
    let line = node_line(node);
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "dotted_name" && child.kind() != "aliased_import" {
            continue;
        }
        let (name, asname) = import_alias(child, src);
        if name.is_empty() {
            continue;
        }
        facts.imports.push(ImportFact {
            target: name.clone(),
            line,
        });
        if !name.contains('.') {
            if let Some(sib) = sibling_module(ctx.rel_path, &name, ctx.known_files) {
                facts.imports.push(ImportFact { target: sib, line });
            }
        }
        let local =
            asname.unwrap_or_else(|| name.split('.').next().unwrap_or(name.as_str()).to_string());
        facts.binds.push((local, name));
    }
}

fn extract_import_from(node: Node<'_>, src: &[u8], facts: &mut FileFacts, ctx: &ExtractCtx<'_>) {
    let line = node_line(node);
    let Some(mod_node) = node.child_by_field_name("module_name") else {
        // `from import x` is invalid; nothing to do.
        return;
    };
    let bases = resolved_from_modules(ctx.rel_path, mod_node, src, ctx.known_files);
    let aliases = import_from_aliases(node, src);
    for base in &bases {
        if !base.is_empty() {
            facts.imports.push(ImportFact {
                target: base.clone(),
                line,
            });
        }
        for (name, asname) in &aliases {
            let target = if base.is_empty() {
                name.clone()
            } else {
                format!("{base}.{name}")
            };
            facts.imports.push(ImportFact {
                target: target.clone(),
                line,
            });
            let local = asname.clone().unwrap_or_else(|| name.clone());
            facts.binds.push((local, target));
        }
    }
}

fn import_from_aliases(node: Node<'_>, src: &[u8]) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "dotted_name" | "aliased_import" => {
                // The module_name field is also a dotted_name / relative_import;
                // skip whatever sits in that field.
                if node.child_by_field_name("module_name").map(|m| m.id()) == Some(child.id()) {
                    continue;
                }
                let (name, asname) = import_alias(child, src);
                if !name.is_empty() {
                    out.push((name, asname));
                }
            }
            "wildcard_import" => out.push(("*".to_string(), None)),
            _ => {}
        }
    }
    out
}

fn import_alias(node: Node<'_>, src: &[u8]) -> (String, Option<String>) {
    match node.kind() {
        "aliased_import" => {
            let name = node
                .child_by_field_name("name")
                .map(|n| node_text(n, src))
                .unwrap_or_default();
            let asname = node.child_by_field_name("alias").map(|n| node_text(n, src));
            (name, asname)
        }
        _ => (node_text(node, src), None),
    }
}

/// Mirror of `_resolved_from_modules` in capability_reachability.py.
fn resolved_from_modules(
    importer: &str,
    mod_node: Node<'_>,
    src: &[u8],
    known_files: &HashSet<String>,
) -> Vec<String> {
    let (level, module) = parse_module_name(mod_node, src);
    let mut bases: Vec<String> = Vec::new();
    if level > 0 {
        let importer_mod = module_name_of(importer);
        let mut parts: Vec<&str> = if importer_mod.is_empty() {
            Vec::new()
        } else {
            importer_mod.split('.').collect()
        };
        let is_init = importer.ends_with("/__init__.py") || importer == "__init__.py";
        if !is_init && !parts.is_empty() {
            parts.pop();
        }
        if level > 1 {
            let cut = (level - 1) as usize;
            let keep = parts.len().saturating_sub(cut);
            parts.truncate(keep);
        }
        let base = parts.join(".");
        let resolved = match (base.is_empty(), module.as_str()) {
            (true, m) => m.to_string(),
            (false, "") => base,
            (false, m) => format!("{base}.{m}"),
        };
        if !resolved.is_empty() {
            bases.push(resolved);
        }
    } else if !module.is_empty() {
        bases.push(module.clone());
        if !module.contains('.') {
            if let Some(sib) = sibling_module(importer, &module, known_files) {
                bases.push(sib);
            }
        }
    }
    bases
}

fn parse_module_name(mod_node: Node<'_>, src: &[u8]) -> (u32, String) {
    match mod_node.kind() {
        "relative_import" => {
            let mut level = 0u32;
            let mut module = String::new();
            let mut cursor = mod_node.walk();
            for child in mod_node.children(&mut cursor) {
                match child.kind() {
                    "import_prefix" => {
                        level = node_text(child, src).chars().filter(|c| *c == '.').count() as u32;
                    }
                    "dotted_name" => module = node_text(child, src),
                    _ => {}
                }
            }
            // Some grammars expose the dots as anonymous "." children instead
            // of a single import_prefix node.
            if level == 0 {
                let mut cursor = mod_node.walk();
                for child in mod_node.children(&mut cursor) {
                    if child.kind() == "." {
                        level += 1;
                    }
                }
            }
            (level, module)
        }
        "import_prefix" => {
            let level = node_text(mod_node, src)
                .chars()
                .filter(|c| *c == '.')
                .count() as u32;
            (level, String::new())
        }
        _ => (0, node_text(mod_node, src)),
    }
}

fn sibling_module(importer: &str, stem: &str, known_files: &HashSet<String>) -> Option<String> {
    let parent = match importer.rfind('/') {
        Some(i) => &importer[..i],
        None => "",
    };
    let sib = if parent.is_empty() {
        format!("{stem}.py")
    } else {
        format!("{parent}/{stem}.py")
    };
    if sib == importer {
        return None;
    }
    if known_files.contains(&sib) {
        Some(module_name_of(&sib))
    } else {
        None
    }
}

pub fn module_name_of(rel_path: &str) -> String {
    let without = rel_path.strip_suffix(".py").unwrap_or(rel_path);
    let mut parts: Vec<&str> = without.split('/').filter(|p| !p.is_empty()).collect();
    if parts.last().copied() == Some("__init__") {
        parts.pop();
    }
    parts.join(".")
}

fn extract_call(node: Node<'_>, src: &[u8], facts: &mut FileFacts) {
    let line = node_line(node);
    let Some(func) = node.child_by_field_name("function") else {
        return;
    };
    match func.kind() {
        "identifier" => {
            let name = node_text(func, src);
            facts.calls.push(CallFact {
                line,
                name: name.clone(),
                qualifier: None,
            });
            if is_subprocess_name(&name) {
                push_subprocess(node, src, line, facts);
            }
        }
        "attribute" => {
            let attr = func
                .child_by_field_name("attribute")
                .map(|n| node_text(n, src))
                .unwrap_or_default();
            let obj = func.child_by_field_name("object");
            if let Some(obj) = obj {
                if obj.kind() == "identifier" {
                    let qualifier = node_text(obj, src);
                    facts.calls.push(CallFact {
                        line,
                        name: attr.clone(),
                        qualifier: Some(qualifier),
                    });
                }
            }
            if is_subprocess_name(&attr) {
                push_subprocess(node, src, line, facts);
            }
        }
        _ => {
            // `foo()()` etc. — Python's analyzer only matches Name / Attribute-of-Name.
            // Still honour a subprocess name if the function is a bare identifier-like
            // attr nested deeper? No: `_is_subprocess_call` only checks the Call's
            // own func. Skip.
        }
    }
}

fn is_subprocess_name(name: &str) -> bool {
    SUBPROCESS_NAMES.iter().any(|n| *n == name)
}

fn push_subprocess(call: Node<'_>, src: &[u8], line: u32, facts: &mut FileFacts) {
    let mut strings = Vec::new();
    collect_str_constants(call, src, &mut strings);
    facts.subprocess.push(SubprocessFact { line, strings });
}

fn collect_str_constants(node: Node<'_>, src: &[u8], out: &mut Vec<String>) {
    match node.kind() {
        "concatenated_string" => {
            let mut parts: Vec<String> = Vec::new();
            let mut ok = true;
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "string" {
                    if let Some(s) = python_str_constant(child, src) {
                        parts.push(s);
                    } else {
                        ok = false;
                    }
                }
            }
            if ok && !parts.is_empty() {
                out.push(parts.concat());
            }
            // Nested interpolations live inside the child strings; walk those
            // for inner Constants the way ast.walk would.
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "string" {
                    collect_str_in_interpolations(child, src, out);
                }
            }
            return;
        }
        "string" => {
            if let Some(s) = python_str_constant(node, src) {
                out.push(s);
            }
            collect_str_in_interpolations(node, src, out);
            return;
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_str_constants(child, src, out);
    }
}

fn collect_str_in_interpolations(string_node: Node<'_>, src: &[u8], out: &mut Vec<String>) {
    let mut cursor = string_node.walk();
    for child in string_node.children(&mut cursor) {
        if child.kind() == "interpolation" {
            collect_str_constants(child, src, out);
        }
    }
}

/// A non-f, non-bytes string's content. `None` if the node is an f-string or
/// bytes literal (Python's `ast.Constant` + `isinstance(..., str)` filter).
fn python_str_constant(node: Node<'_>, src: &[u8]) -> Option<String> {
    let mut contents = String::new();
    let mut is_bytes = false;
    let mut is_f = false;
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "string_start" => {
                let t = node_text(child, src).to_ascii_lowercase();
                if t.contains('b') {
                    is_bytes = true;
                }
                if t.contains('f') {
                    is_f = true;
                }
            }
            "interpolation" => is_f = true,
            "string_content" => contents.push_str(&node_text(child, src)),
            "escape_sequence" => contents.push_str(&node_text(child, src)),
            _ => {}
        }
    }
    if is_bytes || is_f {
        None
    } else {
        Some(contents)
    }
}

/// Index every git-tracked `*.py` file (falling back to a directory walk when
/// `root` is not a git checkout) and emit the aggregated JSON dump.
///
/// Incremental: merkle-diff the py-file leaf set against the previous snapshot
/// in `cache_dir`. Unchanged files reuse cached facts; only dirty files are
/// reparsed.
pub fn collect_reachability_facts(opts: &CollectOptions) -> Result<ReachabilityDump> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&opts.cache_dir)?;
    let cache_path = opts.cache_dir.join("facts.sqlite");
    let conn = open_cache(&cache_path)?;

    let files = list_py_files(&opts.root)?;
    let known: HashSet<String> = files.iter().cloned().collect();
    let blob_map = git_ls_files_blobs(&opts.root).unwrap_or_default();
    let dirty_from_git: HashSet<String> = git_diff_py(&opts.root).unwrap_or_default();

    let mut leaves: BTreeMap<PathBuf, MerkleNode> = BTreeMap::new();
    let mut contents: HashMap<String, String> = HashMap::new();
    let mut hashes: HashMap<String, String> = HashMap::new();
    let mut git_blobs: HashMap<String, String> = HashMap::new();

    let mut need_load: Vec<String> = Vec::new();

    for rel in &files {
        let abs = opts.root.join(rel);
        let git_blob = blob_map.get(rel).cloned().unwrap_or_default();
        git_blobs.insert(rel.clone(), git_blob.clone());
        let cached = load_cached(&conn, rel);
        let on_disk = abs.is_file();
        let is_dirty = dirty_from_git.contains(rel) || (on_disk && git_blob.is_empty());

        if !is_dirty {
            if let Some((cached_blob, cached_hash, _)) = cached.as_ref() {
                if !git_blob.is_empty() && cached_blob == &git_blob {
                    hashes.insert(rel.clone(), cached_hash.clone());
                    leaves.insert(
                        abs.clone(),
                        MerkleNode {
                            path: abs,
                            hash: cached_hash.clone(),
                            kind: MerkleKind::File,
                            size_bytes: 0,
                            children: Vec::new(),
                        },
                    );
                    continue;
                }
            }
        }
        need_load.push(rel.clone());
    }

    let loaded = load_sources(&opts.root, &need_load);
    for (rel, text) in loaded {
        let hash = blake3::hash(text.as_bytes()).to_hex().to_string();
        hashes.insert(rel.clone(), hash.clone());
        contents.insert(rel.clone(), text);
        let abs = opts.root.join(&rel);
        leaves.insert(
            abs.clone(),
            MerkleNode {
                path: abs,
                hash,
                kind: MerkleKind::File,
                size_bytes: 0,
                children: Vec::new(),
            },
        );
    }

    let scanner = Blake3MerkleScanner::new(&opts.root);
    let current = scanner.tree_from_leaves(leaves);
    let prev = load_merkle(&conn);
    let cold = prev.is_none();
    let changeset = match &prev {
        Some(old) => scanner.diff(old, &current)?,
        None => {
            // Cold: every leaf is dirty. We still parse only files we have
            // hashes for; the loop below treats a cache miss as dirty.
            crate::merkle::ChangeSet::default()
        }
    };

    let dirty_abs: HashSet<PathBuf> = if cold {
        HashSet::new()
    } else {
        changeset
            .dirty_paths()
            .into_iter()
            .chain(changeset.added.iter().cloned())
            .collect()
    };

    let mut parsed = 0usize;
    let mut reused = 0usize;
    let mut per_file: BTreeMap<String, FileFacts> = BTreeMap::new();

    for rel in &files {
        let abs = opts.root.join(rel);
        let hash = match hashes.get(rel) {
            Some(h) => h.clone(),
            None => {
                // File listed but unreadable; emit empty facts (Python's OSError → "").
                per_file.insert(rel.clone(), FileFacts::default());
                continue;
            }
        };
        let git_blob = git_blobs.get(rel).cloned().unwrap_or_default();
        let merkle_dirty = cold || dirty_abs.contains(&abs);
        let cache_hit = load_cached(&conn, rel).and_then(|(b, h, facts)| {
            if h == hash || (!git_blob.is_empty() && b == git_blob && !merkle_dirty) {
                Some(facts)
            } else {
                None
            }
        });
        if let Some(facts) = cache_hit {
            if !merkle_dirty || contents.get(rel).is_none() {
                per_file.insert(rel.clone(), facts);
                reused += 1;
                continue;
            }
        }
        let source = contents.get(rel).cloned().unwrap_or_default();
        let facts = extract_python_facts(rel, &source, &known);
        store_cached(&conn, rel, &git_blob, &hash, &facts)?;
        per_file.insert(rel.clone(), facts);
        parsed += 1;
    }

    // Drop facts for files that disappeared.
    prune_missing(&conn, &known)?;
    store_merkle(&conn, &current)?;

    let mut dump = aggregate(files, per_file);
    let nfiles = dump.files.len();
    dump.index = IndexStats {
        merkle_dirty: if cold {
            nfiles
        } else {
            changeset.dirty_paths().len()
        },
        parsed,
        reused,
        files: nfiles,
        elapsed_ms: t0.elapsed().as_millis() as u64,
        cold,
    };
    Ok(dump)
}

fn aggregate(files: Vec<String>, per_file: BTreeMap<String, FileFacts>) -> ReachabilityDump {
    let mut import_sites: BTreeMap<String, Vec<Site>> = BTreeMap::new();
    let mut bound_names: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let mut calls = Vec::new();
    let mut subprocess = Vec::new();
    let mut literals = Vec::new();

    for (file, facts) in &per_file {
        if !facts.binds.is_empty() {
            bound_names.insert(file.clone(), facts.binds.clone());
        }
        for imp in &facts.imports {
            import_sites
                .entry(imp.target.clone())
                .or_default()
                .push(Site {
                    file: file.clone(),
                    line: imp.line,
                    kind: "import".to_string(),
                });
        }
        for c in &facts.calls {
            calls.push(CallSiteDump {
                file: file.clone(),
                line: c.line,
                name: c.name.clone(),
                qualifier: c.qualifier.clone(),
            });
        }
        for s in &facts.subprocess {
            subprocess.push(SubprocessDump {
                file: file.clone(),
                line: s.line,
                strings: s.strings.clone(),
            });
        }
        for lit in &facts.literals {
            literals.push(LiteralDump {
                file: file.clone(),
                line: lit.line,
                token: lit.token.clone(),
            });
        }
    }

    ReachabilityDump {
        schema: FACTS_SCHEMA.to_string(),
        files,
        import_sites,
        bound_names,
        calls,
        subprocess,
        literals,
        index: IndexStats::default(),
    }
}

fn open_cache(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path).map_err(|e| HideError::Storage(e.to_string()))?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         CREATE TABLE IF NOT EXISTS file_facts (
            rel_path TEXT PRIMARY KEY,
            git_blob TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL,
            facts_json TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
         );",
    )
    .map_err(|e| HideError::Storage(e.to_string()))?;
    let ver: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .ok();
    if ver.as_deref() != Some(CACHE_SCHEMA_VERSION) {
        conn.execute_batch("DELETE FROM file_facts; DELETE FROM meta;")
            .map_err(|e| HideError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?1)",
            params![CACHE_SCHEMA_VERSION],
        )
        .map_err(|e| HideError::Storage(e.to_string()))?;
    }
    Ok(conn)
}

fn load_cached(conn: &Connection, rel: &str) -> Option<(String, String, FileFacts)> {
    let row: Option<(String, String, String)> = conn
        .query_row(
            "SELECT git_blob, content_hash, facts_json FROM file_facts WHERE rel_path = ?1",
            params![rel],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .ok();
    let (blob, hash, json) = row?;
    let facts: FileFacts = serde_json::from_str(&json).ok()?;
    Some((blob, hash, facts))
}

fn store_cached(
    conn: &Connection,
    rel: &str,
    git_blob: &str,
    hash: &str,
    facts: &FileFacts,
) -> Result<()> {
    let json = serde_json::to_string(facts)?;
    conn.execute(
        "INSERT INTO file_facts(rel_path, git_blob, content_hash, facts_json)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(rel_path) DO UPDATE SET
           git_blob=excluded.git_blob,
           content_hash=excluded.content_hash,
           facts_json=excluded.facts_json",
        params![rel, git_blob, hash, json],
    )
    .map_err(|e| HideError::Storage(e.to_string()))?;
    Ok(())
}

fn prune_missing(conn: &Connection, known: &HashSet<String>) -> Result<()> {
    let mut stmt = conn
        .prepare("SELECT rel_path FROM file_facts")
        .map_err(|e| HideError::Storage(e.to_string()))?;
    let paths: Vec<String> = stmt
        .query_map([], |r| r.get(0))
        .map_err(|e| HideError::Storage(e.to_string()))?
        .filter_map(|r| r.ok())
        .collect();
    for p in paths {
        if !known.contains(&p) {
            conn.execute("DELETE FROM file_facts WHERE rel_path = ?1", params![p])
                .map_err(|e| HideError::Storage(e.to_string()))?;
        }
    }
    Ok(())
}

fn store_merkle(conn: &Connection, node: &MerkleNode) -> Result<()> {
    let json = serde_json::to_string(node)?;
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('merkle', ?1)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        params![json],
    )
    .map_err(|e| HideError::Storage(e.to_string()))?;
    Ok(())
}

fn load_merkle(conn: &Connection) -> Option<MerkleNode> {
    let json: String = conn
        .query_row("SELECT value FROM meta WHERE key = 'merkle'", [], |r| {
            r.get(0)
        })
        .ok()?;
    serde_json::from_str(&json).ok()
}

fn list_py_files(root: &Path) -> Result<Vec<String>> {
    if let Some(files) = git_ls_py(root) {
        return Ok(files);
    }
    Ok(walk_py(root))
}

fn git_ls_py(root: &Path) -> Option<Vec<String>> {
    let out = std::process::Command::new("git")
        .args(["--no-optional-locks", "ls-files", "-z", "--", "*.py"])
        .current_dir(root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mut files: Vec<String> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .filter_map(|s| String::from_utf8(s.to_vec()).ok())
        .filter(|s| !s.contains("__pycache__"))
        .collect();
    if files.is_empty() && !root.join(".git").exists() {
        return None;
    }
    files.sort();
    Some(files)
}

fn git_ls_files_blobs(root: &Path) -> Option<HashMap<String, String>> {
    let out = std::process::Command::new("git")
        .args(["--no-optional-locks", "ls-files", "-s", "-z", "--", "*.py"])
        .current_dir(root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mut map = HashMap::new();
    for rec in out.stdout.split(|b| *b == 0).filter(|s| !s.is_empty()) {
        let rec = std::str::from_utf8(rec).ok()?;
        // "<mode> <sha> <stage>\t<path>"
        let (meta, path) = rec.split_once('\t')?;
        let mut parts = meta.split_whitespace();
        let _mode = parts.next()?;
        let sha = parts.next()?.to_string();
        map.insert(path.to_string(), sha);
    }
    Some(map)
}

fn git_diff_py(root: &Path) -> Option<HashSet<String>> {
    let out = std::process::Command::new("git")
        .args([
            "--no-optional-locks",
            "diff",
            "--name-only",
            "-z",
            "HEAD",
            "--",
            "*.py",
        ])
        .current_dir(root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        out.stdout
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .filter_map(|s| String::from_utf8(s.to_vec()).ok())
            .collect(),
    )
}

fn walk_py(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let walker = ignore::WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .build();
    for entry in walker.flatten() {
        let path = entry.path();
        let is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
        if !is_file || path.extension().and_then(|e| e.to_str()) != Some("py") {
            continue;
        }
        if let Ok(rel) = path.strip_prefix(root) {
            let s = rel.to_string_lossy().replace('\\', "/");
            if !s.contains("__pycache__") {
                out.push(s);
            }
        }
    }
    out.sort();
    out
}

fn load_sources(root: &Path, rels: &[String]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut missing = Vec::new();
    for rel in rels {
        let abs = root.join(rel);
        match std::fs::read_to_string(&abs) {
            Ok(text) => {
                out.insert(rel.clone(), text);
            }
            Err(_) => missing.push(rel.clone()),
        }
    }
    if !missing.is_empty() {
        for (rel, text) in git_cat_file_batch(root, &missing) {
            out.insert(rel, text);
        }
        for rel in &missing {
            out.entry(rel.clone()).or_insert_with(String::new);
        }
    }
    out
}

fn git_cat_file_batch(root: &Path, rels: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if rels.is_empty() {
        return map;
    }
    let mut input = Vec::new();
    for rel in rels {
        input.extend_from_slice(format!("HEAD:{rel}\n").as_bytes());
    }
    let output = match std::process::Command::new("git")
        .args(["--no-optional-locks", "cat-file", "--batch"])
        .current_dir(root)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(&input)?;
            }
            child.wait_with_output()
        }) {
        Ok(o) if o.status.success() => o.stdout,
        _ => return map,
    };
    let mut idx = 0usize;
    let data = output;
    for rel in rels {
        if idx >= data.len() {
            break;
        }
        let Some(nl) = data[idx..].iter().position(|b| *b == b'\n') else {
            break;
        };
        let header = String::from_utf8_lossy(&data[idx..idx + nl]).into_owned();
        idx += nl + 1;
        if header.contains(" missing") {
            continue;
        }
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() < 3 || parts[1] != "blob" {
            continue;
        }
        let Ok(size) = parts[2].parse::<usize>() else {
            continue;
        };
        if idx + size > data.len() {
            break;
        }
        let blob = &data[idx..idx + size];
        idx += size;
        if idx < data.len() && data[idx] == b'\n' {
            idx += 1;
        }
        let text = String::from_utf8_lossy(blob).into_owned();
        map.insert(rel.clone(), text);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn known(files: &[&str]) -> HashSet<String> {
        files.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn name_call_and_qualified_call() {
        let src = "from pkg.helper import do_it\nimport pkg.helper as h\n\ndef run():\n    do_it()\n    h.do_it()\n    other.do_it()\n";
        let facts = extract_python_facts(
            "pkg/caller.py",
            src,
            &known(&["pkg/caller.py", "pkg/helper.py"]),
        );
        let names: Vec<_> = facts
            .calls
            .iter()
            .map(|c| (c.line, c.name.as_str(), c.qualifier.as_deref()))
            .collect();
        assert!(names.contains(&(5, "do_it", None)), "got {names:?}");
        assert!(names.contains(&(6, "do_it", Some("h"))), "got {names:?}");
        // `other.do_it()` *is* Attribute-of-Name, so it is recorded; Python's
        // assembler then filters it via bound_names. We must still emit it.
        assert!(
            names.contains(&(7, "do_it", Some("other"))),
            "got {names:?}"
        );
        // `a.b.c()` would not be recorded (value is Attribute, not Name).
    }

    #[test]
    fn chained_attr_call_is_not_recorded() {
        let src = "def run():\n    a.b.c()\n";
        let facts = extract_python_facts("x.py", src, &known(&["x.py"]));
        assert!(
            facts.calls.is_empty(),
            "a.b.c() is Attribute-of-Attribute, not a call site: {:?}",
            facts.calls
        );
    }

    #[test]
    fn relative_import_binds_fully_dotted() {
        let src = "from .helper import do_it\n\ndef run():\n    return do_it()\n";
        let facts = extract_python_facts(
            "pkg/caller.py",
            src,
            &known(&["pkg/caller.py", "pkg/helper.py", "pkg/__init__.py"]),
        );
        assert!(
            facts
                .binds
                .iter()
                .any(|(local, full)| local == "do_it" && full == "pkg.helper.do_it"),
            "binds={:?}",
            facts.binds
        );
        assert!(facts
            .imports
            .iter()
            .any(|i| i.target == "pkg.helper" || i.target == "pkg.helper.do_it"));
    }

    #[test]
    fn sibling_import_idiom() {
        let src = "from _common import REPO\n";
        let facts = extract_python_facts(
            "tools/future/foo.py",
            src,
            &known(&["tools/future/foo.py", "tools/future/_common.py"]),
        );
        let targets: Vec<_> = facts.imports.iter().map(|i| i.target.as_str()).collect();
        assert!(
            targets.contains(&"tools.future._common")
                || targets.contains(&"tools.future._common.REPO"),
            "targets={targets:?}"
        );
        assert!(facts
            .binds
            .iter()
            .any(|(_, full)| full == "tools.future._common.REPO"));
    }

    #[test]
    fn subprocess_string_mention_without_launch_is_not_subprocess() {
        let src = "DOC = {\"unrelated_preserved_edit\": \"tools/odyssey_ctl.py\"}\n";
        let facts = extract_python_facts("notes.py", src, &known(&["notes.py"]));
        assert!(facts.subprocess.is_empty(), "{:?}", facts.subprocess);
    }

    #[test]
    fn subprocess_run_records_path_string() {
        let src = "import subprocess\nsubprocess.run([\"python3\", \"tools/odyssey_ctl.py\", \"cycle\"])\n";
        let facts = extract_python_facts("launcher.py", src, &known(&["launcher.py"]));
        assert_eq!(facts.subprocess.len(), 1);
        assert!(
            facts.subprocess[0]
                .strings
                .iter()
                .any(|s| s.contains("tools/odyssey_ctl.py")),
            "{:?}",
            facts.subprocess[0].strings
        );
    }

    #[test]
    fn tool_dispatch_not_bare_set() {
        let src = "ALL_TOOLS = {\"git.status\", \"fs.read\"}\nCATALOG = [{\"label\": \"x\", \"tool\": \"git.status\", \"arguments\": {}}]\n";
        let facts = extract_python_facts("gate.py", src, &known(&["gate.py"]));
        let tokens: Vec<_> = facts.literals.iter().map(|l| l.token.as_str()).collect();
        assert_eq!(tokens, vec!["git.status"]);
    }

    #[test]
    fn module_import_is_not_a_call() {
        let src = "import tools.future.widget\nfrom tools.future.widget import gadget\n";
        let facts = extract_python_facts(
            "caller.py",
            src,
            &known(&["caller.py", "tools/future/widget.py"]),
        );
        assert!(
            facts.calls.is_empty(),
            "import is not a call: {:?}",
            facts.calls
        );
        assert!(!facts.imports.is_empty());
    }

    #[test]
    fn incremental_reuse_on_unchanged_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/a.py"), "def alpha():\n    return 1\n").unwrap();
        fs::write(
            root.join("src/b.py"),
            "from src.a import alpha\n\ndef beta():\n    return alpha()\n",
        )
        .unwrap();
        let cache = root.join("cache");
        let opts = CollectOptions::new(root).with_cache_dir(&cache);
        let first = collect_reachability_facts(&opts).unwrap();
        assert!(first.index.cold);
        assert_eq!(first.index.parsed, 2);
        assert!(first
            .calls
            .iter()
            .any(|c| c.name == "alpha" && c.qualifier.is_none()));
        let second = collect_reachability_facts(&opts).unwrap();
        assert!(!second.index.cold);
        assert_eq!(second.index.reused, 2, "warm run must reuse both files");
        assert_eq!(second.index.parsed, 0);
        assert_eq!(first.calls, second.calls);
        assert_eq!(first.import_sites, second.import_sites);
    }

    #[test]
    fn one_file_edit_reparses_only_that_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::write(root.join("a.py"), "def alpha():\n    return 1\n").unwrap();
        fs::write(root.join("b.py"), "def beta():\n    return 2\n").unwrap();
        let cache = root.join("cache");
        let opts = CollectOptions::new(root).with_cache_dir(&cache);
        collect_reachability_facts(&opts).unwrap();
        fs::write(root.join("a.py"), "def alpha():\n    return helper()\n").unwrap();
        let after = collect_reachability_facts(&opts).unwrap();
        assert_eq!(after.index.parsed, 1, "only the edited file is reparsed");
        assert_eq!(after.index.reused, 1);
        assert!(after.calls.iter().any(|c| c.name == "helper"));
    }
}
