//! hide-compat: configuration-compatibility readers.
//!
//! HIDE is a local-first IDE. A repository that was set up for another coding
//! agent (Claude Code, in particular) carries a tree of configuration files:
//! `CLAUDE.md` memory with `@imports`, `.claude/rules`, `settings.json` across
//! several scopes, subagent and skill definitions, and `.mcp.json` server
//! declarations. This crate reads all of that WITHOUT running any model. It is
//! pure parsing plus precedence resolution so a repo migrates to HIDE with
//! minimal changes (Bible sec 58 / Phase 11).
//!
//! Every reader takes a [`layout::Layout`], which names the filesystem locations
//! for each scope. Nothing here touches process globals beyond the optional
//! `Layout::discover` helper, so the whole crate is deterministic and fully
//! testable against a tempdir fixture.
//!
//! The two precedence rules that most often trip people up are kept explicit:
//! - Settings permission arrays MERGE across scopes and `deny` wins a decision;
//!   scalar settings resolve Managed > CLI > Local > Project > User.
//! - Instruction layers use a SEPARATE precedence Managed > User > Project >
//!   Local, applied read-last-wins.

pub use error::{CompatError, Result};
pub use layout::Layout;

/// The complete compatibility view of a repository.
#[derive(Debug, Clone)]
pub struct CompatConfig {
    pub memory: claude_md::MemoryTree,
    pub rules: Vec<rules::Rule>,
    pub settings: settings::ResolvedSettings,
    pub agents: Vec<agents::Agent>,
    pub skills: Vec<skills::Skill>,
    pub mcp: mcp::McpConfig,
}

impl CompatConfig {
    /// Read and resolve every compatibility source for a layout.
    ///
    /// `cli` optionally supplies the CLI settings scope (parsed command-line
    /// flags), which sits just under Managed in scalar precedence.
    pub fn load(layout: &Layout, cli: Option<settings::RawSettings>) -> Result<CompatConfig> {
        let settings = settings::load(layout, cli)?;
        let excludes = settings.excludes_glob_set()?;
        let memory = claude_md::discover(layout, excludes.as_ref());
        let rules = rules::discover(layout);
        let agents = agents::discover(layout);
        let skills = skills::discover(layout);
        let mcp = mcp::load(layout)?;

        Ok(CompatConfig {
            memory,
            rules,
            settings,
            agents,
            skills,
            mcp,
        })
    }

    /// The instruction sources injected at launch, in application order:
    /// memory files (root first, more specific last) followed by un-scoped
    /// rules. Scoped rules and lazy subtree memory are excluded (they attach to
    /// matching-file reads instead).
    pub fn launch_instruction_order(&self) -> Vec<std::path::PathBuf> {
        let mut order: Vec<std::path::PathBuf> = self.memory.launch_order();
        for rule in &self.rules {
            if rule.loads_at_launch() {
                order.push(rule.path.clone());
            }
        }
        order
    }
}

// --- inlined compat/agents.rs ---
pub mod agents {
//! Subagent definitions: `.claude/agents/*.md` and `~/.claude/agents/*.md`.
//!
//! Each agent is a markdown file with YAML frontmatter. `disallowedTools` is
//! applied before `tools` when computing the effective tool set, so a tool that
//! is both allowed and disallowed is removed. `model` defaults to `inherit`.

use std::path::{Path, PathBuf};

use crate::compat::frontmatter::{self, Frontmatter};
use crate::compat::layout::Layout;

/// A parsed subagent definition.
#[derive(Debug, Clone)]
pub struct Agent {
    pub path: PathBuf,
    pub name: String,
    pub description: Option<String>,
    pub tools: Vec<String>,
    pub disallowed_tools: Vec<String>,
    /// Defaults to "inherit".
    pub model: String,
    pub skills: Vec<String>,
    pub mcp: Vec<String>,
    pub hooks: Vec<String>,
    pub memory: Vec<String>,
    pub permissions: Vec<String>,
    /// The agent's system-prompt body (frontmatter stripped).
    pub body: String,
}

impl Agent {
    /// Tools the agent may actually use: `tools` minus `disallowedTools`.
    /// disallowed is applied before allow.
    pub fn effective_tools(&self) -> Vec<String> {
        self.tools
            .iter()
            .filter(|t| !self.disallowed_tools.contains(t))
            .cloned()
            .collect()
    }

    /// Whether `model` is the inherit sentinel.
    pub fn inherits_model(&self) -> bool {
        self.model.eq_ignore_ascii_case("inherit")
    }

    /// Whether this profile permits `tool`. This is the gate predicate: deny
    /// wins, and an EMPTY `tools` list means "inherit every tool" (omitting the
    /// `tools:` key declines to narrow the set, it does not strip the agent of
    /// tools).
    ///
    /// [`Self::effective_tools`] answers a different question, the explicitly
    /// listed set, and therefore returns empty for an inherit-all profile. A
    /// caller that gates on it would read that empty vec as deny-all (or, worse,
    /// as allow-all while silently dropping `disallowedTools`), so gate on this
    /// instead.
    pub fn allows_tool(&self, tool: &str) -> bool {
        if self.disallowed_tools.iter().any(|t| t == tool) {
            return false;
        }
        self.tools.is_empty() || self.tools.iter().any(|t| t == tool)
    }
}

/// Parse a single agent markdown file.
pub fn parse(path: &Path) -> Option<Agent> {
    let raw = std::fs::read_to_string(path).ok()?;
    let (fm, body) = frontmatter::split(&raw);
    let fm = fm.unwrap_or_default();
    Some(from_frontmatter(path, &fm, body))
}

fn from_frontmatter(path: &Path, fm: &Frontmatter, body: String) -> Agent {
    let name = fm
        .str("name")
        .unwrap_or_else(|| stem(path).to_string());
    Agent {
        path: path.to_path_buf(),
        name,
        description: fm.str("description"),
        tools: fm.list("tools"),
        disallowed_tools: fm.list("disallowedTools"),
        model: fm.str("model").unwrap_or_else(|| "inherit".to_string()),
        skills: fm.list("skills"),
        mcp: fm.list("mcp"),
        hooks: fm.list("hooks"),
        memory: fm.list("memory"),
        permissions: fm.list("permissions"),
        body,
    }
}

fn stem(path: &Path) -> &str {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("agent")
}

/// Discover all agents. Project agents (`<root>/.claude/agents`) take precedence
/// over user agents (`~/.claude/agents`) of the same name. Returns them sorted
/// by name for determinism.
pub fn discover(layout: &Layout) -> Vec<Agent> {
    let mut by_name: std::collections::BTreeMap<String, Agent> =
        std::collections::BTreeMap::new();

    // User first, then project overrides by name.
    for dir in [
        layout.home.join(".claude").join("agents"),
        layout.repo_root.join(".claude").join("agents"),
    ] {
        for agent in parse_dir(&dir) {
            by_name.insert(agent.name.clone(), agent);
        }
    }

    by_name.into_values().collect()
}

fn parse_dir(dir: &Path) -> Vec<Agent> {
    if !dir.exists() {
        return Vec::new();
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file()
            && entry.path().extension().map(|e| e == "md").unwrap_or(false)
        {
            files.push(entry.into_path());
        }
    }
    files.sort();
    files.iter().filter_map(|p| parse(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disallowed_applied_before_allow() {
        let fm = frontmatter::parse_block(
            "name: reviewer\ntools: [Read, Write, Bash]\ndisallowedTools: [Bash]\n",
        );
        let agent = from_frontmatter(Path::new("/x/reviewer.md"), &fm, String::new());
        assert_eq!(agent.effective_tools(), vec!["Read", "Write"]);
    }
    #[test]
    fn empty_tool_list_inherits_all_and_deny_still_wins() {
        let fm = frontmatter::parse_block("name: reviewer\ndisallowedTools: [Bash]\n");
        let agent = from_frontmatter(Path::new("/x/reviewer.md"), &fm, String::new());
        assert!(agent.effective_tools().is_empty());
        assert!(agent.allows_tool("Read"));
        assert!(!agent.allows_tool("Bash"));
    }
    #[test]
    fn explicit_tool_list_is_a_closed_set() {
        let fm = frontmatter::parse_block("name: r\ntools: [Read]\ndisallowedTools: [Bash]\n");
        let agent = from_frontmatter(Path::new("/x/r.md"), &fm, String::new());
        assert!(agent.allows_tool("Read"));
        assert!(!agent.allows_tool("Write"));
        assert!(!agent.allows_tool("Bash"));
    }
    #[test]
    fn model_defaults_to_inherit() {
        let fm = frontmatter::parse_block("name: a\n");
        let agent = from_frontmatter(Path::new("/x/a.md"), &fm, String::new());
        assert!(agent.inherits_model());
    }
}
}


// --- inlined compat/claude_md.rs ---
pub mod claude_md {
//! CLAUDE.md tree discovery, HTML comment stripping, and `@import` resolution.
//!
//! Discovery walks the directory chain from the repo root down to the working
//! directory (root first, so more specific memory is injected last) and collects
//! `CLAUDE.md`, `.claude/CLAUDE.md`, and `CLAUDE.local.md` per directory. A user
//! global `~/.claude/CLAUDE.md` sorts ahead of the project chain. CLAUDE.md files
//! that live in subdirectories *below* the working directory are recorded but
//! marked lazy: they are meant to be injected only when a file under that subtree
//! is read, not at launch.
//!
//! `@path` imports inline another file's (comment-stripped) content. Imports may
//! be relative to the importing file, absolute, or `~/`-anchored, and recurse up
//! to depth 4. An `@` inside inline backticks or a fenced code block is left
//! alone. The first import that resolves outside the repo root is flagged as
//! requiring approval.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use globset::GlobSet;

use crate::compat::layout::Layout;

/// Maximum `@import` recursion depth (levels below the launch-injected file).
pub const MAX_IMPORT_DEPTH: usize = 4;

/// Where a memory file sits in the scope hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// `~/.claude/CLAUDE.md`.
    UserGlobal,
    /// `<dir>/CLAUDE.md`.
    Project,
    /// `<dir>/.claude/CLAUDE.md`.
    DotClaude,
    /// `<dir>/CLAUDE.local.md`.
    Local,
    /// A `CLAUDE.md` in a subdirectory below the working directory.
    Subtree,
}

/// A single record of one `@import`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportRecord {
    /// The raw token as written (without the leading `@`).
    pub spec: String,
    /// The resolved absolute path.
    pub resolved: PathBuf,
    /// Depth at which this import was encountered (1 = imported by a launch file).
    pub depth: usize,
    /// Whether the target resolved outside the repo root.
    pub external: bool,
    /// Whether this import was flagged for user approval (the first external one).
    pub approval_required: bool,
    /// Whether the file existed and was inlined (false if missing or past cutoff).
    pub inlined: bool,
}

/// One discovered memory file.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub path: PathBuf,
    pub kind: MemoryKind,
    /// Lazy entries are not injected at launch; they attach to a subtree.
    pub lazy: bool,
    /// Raw file bytes as read.
    pub raw: String,
    /// Injection-ready content: HTML comments stripped, imports inlined.
    pub injected: String,
    /// Every import encountered while resolving this entry (any depth).
    pub imports: Vec<ImportRecord>,
}

/// The full discovered memory tree for a layout.
#[derive(Debug, Clone, Default)]
pub struct MemoryTree {
    pub entries: Vec<MemoryEntry>,
}

impl MemoryTree {
    /// Launch-injected entries in order (root first, more specific last). Lazy
    /// subtree entries are excluded.
    pub fn launch_entries(&self) -> Vec<&MemoryEntry> {
        self.entries.iter().filter(|e| !e.lazy).collect()
    }

    /// Paths of the launch-injected entries, in order. Handy for order assertions.
    pub fn launch_order(&self) -> Vec<PathBuf> {
        self.launch_entries().iter().map(|e| e.path.clone()).collect()
    }

    /// Lazy subtree entries (injected on demand when their subtree is read).
    pub fn lazy_entries(&self) -> Vec<&MemoryEntry> {
        self.entries.iter().filter(|e| e.lazy).collect()
    }

    /// Whether any import across the tree was flagged for approval.
    pub fn approval_required(&self) -> bool {
        self.entries
            .iter()
            .flat_map(|e| e.imports.iter())
            .any(|i| i.approval_required)
    }
}

/// Discover the CLAUDE.md tree. `excludes` is an optional compiled glob set from
/// `claudeMdExcludes` in settings; matching files are skipped entirely.
pub fn discover(layout: &Layout, excludes: Option<&GlobSet>) -> MemoryTree {
    let mut entries: Vec<MemoryEntry> = Vec::new();
    let mut approval_used = false;

    let excluded = |p: &Path| -> bool {
        match excludes {
            Some(set) => {
                // Match on both the absolute path and the path relative to the
                // repo root so simple globs like `packages/**/CLAUDE.md` work.
                let rel = p
                    .strip_prefix(&layout.repo_root)
                    .unwrap_or(p);
                set.is_match(p) || set.is_match(rel)
            }
            None => false,
        }
    };

    let push = |entries: &mut Vec<MemoryEntry>,
                    approval_used: &mut bool,
                    path: PathBuf,
                    kind: MemoryKind,
                    lazy: bool| {
        if excluded(&path) {
            return;
        }
        let raw = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => return,
        };
        let stripped = strip_html_comments(&raw);
        let mut imports = Vec::new();
        let injected = if lazy {
            // Lazy entries are not resolved until injected; keep stripped body.
            stripped.clone()
        } else {
            let mut visited = HashSet::new();
            visited.insert(canonical_key(&path));
            resolve_imports(
                &stripped,
                &path,
                layout,
                1,
                &mut visited,
                &mut imports,
                approval_used,
            )
        };
        entries.push(MemoryEntry {
            path,
            kind,
            lazy,
            raw,
            injected,
            imports,
        });
    };

    // 1. User global memory, least specific, injected first.
    let user_global = layout.home.join(".claude").join("CLAUDE.md");
    push(
        &mut entries,
        &mut approval_used,
        user_global,
        MemoryKind::UserGlobal,
        false,
    );

    // 2. The repo-root-down-to-cwd chain, root first.
    for dir in layout.dir_chain_root_first() {
        push(
            &mut entries,
            &mut approval_used,
            dir.join("CLAUDE.md"),
            MemoryKind::Project,
            false,
        );
        push(
            &mut entries,
            &mut approval_used,
            dir.join(".claude").join("CLAUDE.md"),
            MemoryKind::DotClaude,
            false,
        );
        push(
            &mut entries,
            &mut approval_used,
            dir.join("CLAUDE.local.md"),
            MemoryKind::Local,
            false,
        );
    }

    // 3. Subtree CLAUDE.md files below cwd, marked lazy. Sorted for determinism.
    let mut subtree: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(&layout.cwd)
        .min_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() && entry.file_name() == "CLAUDE.md" {
            subtree.push(entry.into_path());
        }
    }
    subtree.sort();
    for path in subtree {
        push(
            &mut entries,
            &mut approval_used,
            path,
            MemoryKind::Subtree,
            true,
        );
    }

    MemoryTree { entries }
}

fn canonical_key(p: &Path) -> PathBuf {
    p.to_path_buf()
}

/// Resolve `@import` tokens in `content`, returning content with imports inlined.
#[allow(clippy::too_many_arguments)]
fn resolve_imports(
    content: &str,
    importing_file: &Path,
    layout: &Layout,
    depth: usize,
    visited: &mut HashSet<PathBuf>,
    records: &mut Vec<ImportRecord>,
    approval_used: &mut bool,
) -> String {
    let importing_dir = importing_file.parent().unwrap_or(Path::new("."));
    let mut out = String::with_capacity(content.len());
    let mut fence: Option<String> = None;

    for line in content.lines() {
        let trimmed = line.trim_start();
        // Fenced code block toggles: a run of >=3 backticks or tildes.
        if let Some(marker) = fence_marker(trimmed) {
            match &fence {
                None => fence = Some(marker),
                Some(open) if marker.starts_with(open.chars().next().unwrap()) => fence = None,
                Some(_) => {}
            }
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if fence.is_some() {
            // Inside a code fence: imports are inert.
            out.push_str(line);
            out.push('\n');
            continue;
        }

        // Scan the line for import tokens outside inline backticks.
        let tokens = scan_line_imports(line);
        if tokens.is_empty() {
            out.push_str(line);
            out.push('\n');
            continue;
        }

        // The line has at least one import. Emit the line, then inline each
        // imported file after it (Claude Code injects imported memory alongside).
        out.push_str(line);
        out.push('\n');
        for spec in tokens {
            let resolved = resolve_import_path(&spec, importing_dir, layout);
            let external = !resolved.starts_with(&layout.repo_root);
            let approval_required = external && !*approval_used;
            if approval_required {
                *approval_used = true;
            }

            let mut inlined = false;
            if depth <= MAX_IMPORT_DEPTH {
                let key = canonical_key(&resolved);
                if !visited.contains(&key) {
                    if let Ok(raw) = std::fs::read_to_string(&resolved) {
                        visited.insert(key);
                        let stripped = strip_html_comments(&raw);
                        let nested = resolve_imports(
                            &stripped,
                            &resolved,
                            layout,
                            depth + 1,
                            visited,
                            records,
                            approval_used,
                        );
                        out.push_str(&nested);
                        if !nested.ends_with('\n') {
                            out.push('\n');
                        }
                        inlined = true;
                    }
                }
            }

            records.push(ImportRecord {
                spec,
                resolved,
                depth,
                external,
                approval_required,
                inlined,
            });
        }
    }

    out
}

/// Resolve one import spec to an absolute path.
fn resolve_import_path(spec: &str, importing_dir: &Path, layout: &Layout) -> PathBuf {
    if let Some(rest) = spec.strip_prefix("~/") {
        return layout.home.join(rest);
    }
    if spec == "~" {
        return layout.home.clone();
    }
    let p = Path::new(spec);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    importing_dir.join(spec)
}

/// Return the fence marker string if `trimmed` opens or closes a code fence.
fn fence_marker(trimmed: &str) -> Option<String> {
    let first = trimmed.chars().next()?;
    if first != '`' && first != '~' {
        return None;
    }
    let run: String = trimmed.chars().take_while(|&c| c == first).collect();
    if run.len() >= 3 {
        Some(run)
    } else {
        None
    }
}

/// Find `@import` tokens on a single line, skipping any inside inline backticks.
fn scan_line_imports(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    let mut in_code = false;
    let mut prev_boundary = true; // start of line is a boundary
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            in_code = !in_code;
            prev_boundary = true;
            i += 1;
            continue;
        }
        if c == '@' && !in_code && prev_boundary {
            // Collect the path token: everything up to whitespace or backtick.
            let mut j = i + 1;
            while j < chars.len() {
                let cj = chars[j];
                if cj.is_whitespace() || cj == '`' {
                    break;
                }
                j += 1;
            }
            let mut token: String = chars[i + 1..j].iter().collect();
            // Trim trailing punctuation that is unlikely to be part of a path.
            while let Some(last) = token.chars().last() {
                if matches!(last, ',' | '.' | ';' | ':' | ')' | ']' | '!' | '?')
                    && !token.ends_with(".md")
                {
                    token.pop();
                } else {
                    break;
                }
            }
            if !token.is_empty() {
                out.push(token);
            }
            i = j;
            prev_boundary = true;
            continue;
        }
        prev_boundary = c.is_whitespace() || c == '(' || c == '[';
        i += 1;
    }
    out
}

/// Strip block-level HTML comments (`<!-- ... -->`) from markdown, preserving any
/// that appear inside fenced code blocks. Multi-line comments are supported.
pub fn strip_html_comments(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut fence: Option<char> = None;
    let mut in_comment = false;

    for line in content.lines() {
        let trimmed = line.trim_start();

        // Fence handling first: never strip inside code.
        if fence.is_none() && !in_comment {
            if let Some(marker) = fence_marker(trimmed) {
                fence = Some(marker.chars().next().unwrap());
                out.push_str(line);
                out.push('\n');
                continue;
            }
        } else if let Some(f) = fence {
            if let Some(marker) = fence_marker(trimmed) {
                if marker.starts_with(f) {
                    fence = None;
                }
            }
            out.push_str(line);
            out.push('\n');
            continue;
        }

        if fence.is_some() {
            out.push_str(line);
            out.push('\n');
            continue;
        }

        // Outside code: strip comments, handling multi-line spans.
        let mut rest = line;
        let mut kept = String::new();
        loop {
            if in_comment {
                if let Some(end) = rest.find("-->") {
                    rest = &rest[end + 3..];
                    in_comment = false;
                } else {
                    // Comment continues past end of line; drop the remainder.
                    break;
                }
            } else if let Some(start) = rest.find("<!--") {
                kept.push_str(&rest[..start]);
                rest = &rest[start + 4..];
                in_comment = true;
            } else {
                kept.push_str(rest);
                break;
            }
        }

        // Only emit the line if it retained non-whitespace, or was originally
        // blank. This drops lines that were nothing but a comment.
        if !kept.trim().is_empty() || line.trim().is_empty() {
            out.push_str(&kept);
            out.push('\n');
        } else if !kept.is_empty() {
            // Kept only whitespace from a comment-only line: drop it.
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strips_block_comment_but_keeps_fenced() {
        let doc = "before\n<!-- secret -->\nafter\n```\n<!-- kept in code -->\n```\n";
        let out = strip_html_comments(doc);
        assert!(!out.contains("secret"));
        assert!(out.contains("before"));
        assert!(out.contains("after"));
        assert!(out.contains("<!-- kept in code -->"));
    }
    #[test]
    fn strips_multiline_comment() {
        let doc = "a\n<!-- line1\nline2 -->b\nc\n";
        let out = strip_html_comments(doc);
        assert!(!out.contains("line1"));
        assert!(!out.contains("line2"));
        assert!(out.contains('a'));
        assert!(out.contains('b'));
        assert!(out.contains('c'));
    }
    #[test]
    fn scan_skips_inline_backtick() {
        let tokens = scan_line_imports("see @real/file.md but not `@fake/file.md`");
        assert_eq!(tokens, vec!["real/file.md".to_string()]);
    }
    #[test]
    fn scan_ignores_mid_word_at() {
        let tokens = scan_line_imports("email me@example.com is not an import");
        assert!(tokens.is_empty());
    }
    #[test]
    fn fence_marker_detects_backticks_and_tildes() {
        assert_eq!(fence_marker("```rust"), Some("```".to_string()));
        assert_eq!(fence_marker("~~~"), Some("~~~".to_string()));
        assert_eq!(fence_marker("``inline"), None);
        assert_eq!(fence_marker("text"), None);
    }
}
}


// --- inlined compat/error.rs ---
pub mod error {
use thiserror::Error;

/// Errors surfaced by the compatibility readers. Parsing is intentionally
/// lenient (a malformed optional file is skipped, not fatal); these variants
/// cover the cases where the caller genuinely cannot proceed.
#[derive(Debug, Error)]
pub enum CompatError {
    #[error("io error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("json parse error in {path}: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid glob {glob:?}: {source}")]
    Glob {
        glob: String,
        #[source]
        source: globset::Error,
    },

    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, CompatError>;

impl CompatError {
    pub fn msg(m: impl Into<String>) -> Self {
        CompatError::Message(m.into())
    }
}
}


// --- inlined compat/frontmatter.rs ---
pub mod frontmatter {
//! A deliberately small YAML-subset reader for markdown frontmatter.
//!
//! Agent and skill definitions carry a `---` delimited frontmatter block with a
//! constrained shape: scalar keys, booleans, and lists (either block form with
//! `- item` lines or inline `[a, b, c]`). We do not want a full YAML dependency
//! for a model-free compatibility layer, so this parser handles exactly that
//! subset and nothing more. It is deterministic and total: malformed input never
//! panics, it simply yields whatever keys it could recognise.

use std::collections::BTreeMap;

/// A recognised frontmatter value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Scalar(String),
    Bool(bool),
    List(Vec<String>),
    Null,
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Scalar(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// A list view. A bare scalar is treated as a one-element list so callers
    /// that accept either `tools: Read` or `tools: [Read, Write]` behave the same.
    pub fn as_list(&self) -> Vec<String> {
        match self {
            Value::List(v) => v.clone(),
            Value::Scalar(s) => vec![s.clone()],
            _ => Vec::new(),
        }
    }
}

/// A parsed frontmatter block. Keys are ordered for stable iteration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Frontmatter {
    pub map: BTreeMap<String, Value>,
}

impl Frontmatter {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }

    pub fn str(&self, key: &str) -> Option<String> {
        self.map.get(key).and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        self.map.get(key).and_then(|v| v.as_bool())
    }

    pub fn list(&self, key: &str) -> Vec<String> {
        self.map.get(key).map(|v| v.as_list()).unwrap_or_default()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }
}

/// Split a markdown document into its optional frontmatter block and the body
/// that follows it. If there is no leading `---` fence the whole document is
/// returned as the body and the frontmatter is `None`.
pub fn split(content: &str) -> (Option<Frontmatter>, String) {
    // A frontmatter block must be the very first line of the file. Allow a BOM
    // and trailing whitespace on the fence line, nothing else before it.
    let stripped = content.strip_prefix('\u{feff}').unwrap_or(content);
    let mut lines = stripped.lines();
    let first = match lines.next() {
        Some(l) => l,
        None => return (None, String::new()),
    };
    if first.trim() != "---" {
        return (None, content.to_string());
    }

    let mut block = String::new();
    let mut closed = false;
    let mut body = String::new();
    let mut in_body = false;
    for line in lines {
        if in_body {
            body.push_str(line);
            body.push('\n');
            continue;
        }
        if line.trim() == "---" {
            closed = true;
            in_body = true;
            continue;
        }
        block.push_str(line);
        block.push('\n');
    }

    if !closed {
        // Unterminated fence: treat the whole thing as body, no frontmatter.
        return (None, content.to_string());
    }

    (Some(parse_block(&block)), body)
}

/// Parse just a frontmatter block (already stripped of its `---` fences).
pub fn parse_block(text: &str) -> Frontmatter {
    let mut map: BTreeMap<String, Value> = BTreeMap::new();
    let mut cur_key: Option<String> = None;
    let mut cur_list: Vec<String> = Vec::new();

    // Flush a pending block-list key into the map.
    fn flush(map: &mut BTreeMap<String, Value>, key: Option<String>, list: &mut Vec<String>) {
        if let Some(k) = key {
            if list.is_empty() {
                map.insert(k, Value::Null);
            } else {
                map.insert(k, Value::List(std::mem::take(list)));
            }
        }
    }

    for raw in text.lines() {
        let line = strip_inline_comment(raw);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let indented = raw
            .chars()
            .next()
            .map(|c| c == ' ' || c == '\t')
            .unwrap_or(false);

        if indented && trimmed.starts_with('-') {
            // Block-list item belonging to the current key.
            if cur_key.is_some() {
                let item = trimmed[1..].trim();
                if !item.is_empty() {
                    cur_list.push(unquote(item));
                }
                continue;
            }
            // A dash with no owning key: ignore.
            continue;
        }

        // A new key starts here; flush any pending block list first.
        flush(&mut map, cur_key.take(), &mut cur_list);

        let colon = match trimmed.find(':') {
            Some(i) => i,
            None => continue,
        };
        let key = trimmed[..colon].trim().to_string();
        if key.is_empty() {
            continue;
        }
        let rest = trimmed[colon + 1..].trim();

        if rest.is_empty() {
            // Possibly a block list; wait for indented `- ` lines.
            cur_key = Some(key);
        } else if rest.starts_with('[') && rest.ends_with(']') {
            let inner = &rest[1..rest.len() - 1];
            let items: Vec<String> = inner
                .split(',')
                .map(|s| unquote(s.trim()))
                .filter(|s| !s.is_empty())
                .collect();
            map.insert(key, Value::List(items));
        } else {
            map.insert(key, scalar_value(rest));
        }
    }

    flush(&mut map, cur_key.take(), &mut cur_list);
    Frontmatter { map }
}

fn scalar_value(rest: &str) -> Value {
    match rest {
        "true" | "True" | "yes" => Value::Bool(true),
        "false" | "False" | "no" => Value::Bool(false),
        "null" | "~" => Value::Null,
        _ => Value::Scalar(unquote(rest)),
    }
}

/// Strip a trailing `# comment` that is not inside a quoted string. Matching
/// YAML, a `#` is a comment only when it follows whitespace (or starts the
/// line); a `#` glued to non-whitespace (e.g. `a#b`) is kept. A hex value must
/// therefore be quoted (`color: "#ffffff"`) to survive, exactly as in YAML.
fn strip_inline_comment(line: &str) -> String {
    let bytes: Vec<char> = line.chars().collect();
    let mut in_single = false;
    let mut in_double = false;
    let mut prev_ws = true; // start-of-line counts as preceding whitespace
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' if !in_single && !in_double && prev_ws => {
                return bytes[..i].iter().collect::<String>();
            }
            _ => {}
        }
        prev_ws = c == ' ' || c == '\t';
    }
    line.to_string()
}

/// Remove a single pair of matching surrounding quotes if present.
fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 {
        let bytes = s.as_bytes();
        let first = bytes[0];
        let last = bytes[s.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return s[1..s.len() - 1].to_string();
        }
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn splits_frontmatter_and_body() {
        let doc = "---\nname: foo\n---\nbody line\nmore\n";
        let (fm, body) = split(doc);
        let fm = fm.expect("frontmatter present");
        assert_eq!(fm.str("name").as_deref(), Some("foo"));
        assert_eq!(body, "body line\nmore\n");
    }
    #[test]
    fn no_frontmatter_returns_whole_body() {
        let doc = "just a body\nno fence\n";
        let (fm, body) = split(doc);
        assert!(fm.is_none());
        assert_eq!(body, doc);
    }
    #[test]
    fn parses_inline_and_block_lists() {
        let block = "tools: [Read, Write]\nskills:\n  - a\n  - b\n";
        let fm = parse_block(block);
        assert_eq!(fm.list("tools"), vec!["Read", "Write"]);
        assert_eq!(fm.list("skills"), vec!["a", "b"]);
    }
    #[test]
    fn parses_booleans_and_quotes() {
        let block = "user-invocable: true\ndisable-model-invocation: false\ndescription: \"hello world\"\n";
        let fm = parse_block(block);
        assert_eq!(fm.bool("user-invocable"), Some(true));
        assert_eq!(fm.bool("disable-model-invocation"), Some(false));
        assert_eq!(fm.str("description").as_deref(), Some("hello world"));
    }
    #[test]
    fn empty_key_becomes_null_not_list() {
        let block = "memory:\nname: x\n";
        let fm = parse_block(block);
        assert_eq!(fm.get("memory"), Some(&Value::Null));
        assert_eq!(fm.str("name").as_deref(), Some("x"));
    }
    #[test]
    fn strips_inline_comment_and_quoted_hash_survives() {
        let block = "quoted: \"#ffffff\"\nname: foo # trailing\nglued: a#b\nbare: #ffffff\n";
        let fm = parse_block(block);
        assert_eq!(fm.str("quoted").as_deref(), Some("#ffffff"));
        assert_eq!(fm.str("name").as_deref(), Some("foo"));
        assert_eq!(fm.str("glued").as_deref(), Some("a#b"));
        assert_eq!(fm.get("bare"), Some(&Value::Null));
    }
}
}


// --- inlined compat/layout.rs ---
pub mod layout {
//! Scope layout: the set of filesystem locations the compatibility readers scan.
//!
//! Every reader takes a `&Layout` rather than touching process globals, which
//! keeps the whole crate deterministic and lets tests point every scope at a
//! tempdir. `Layout::discover` is a convenience for real use that fills the
//! scopes from `$HOME` and the nearest enclosing git repo.

use std::path::{Path, PathBuf};

/// Filesystem locations for every configuration scope.
#[derive(Debug, Clone)]
pub struct Layout {
    /// The working directory the agent was launched from.
    pub cwd: PathBuf,
    /// The repository root (chain of CLAUDE.md files stops here going up).
    pub repo_root: PathBuf,
    /// The user home directory (`~/.claude`, `~/.claude.json` live under it).
    pub home: PathBuf,
    /// Optional managed (enterprise) settings.json path.
    pub managed_settings: Option<PathBuf>,
    /// Optional managed MCP config path.
    pub managed_mcp: Option<PathBuf>,
}

impl Layout {
    /// Construct an explicit layout. Prefer this in tests.
    pub fn new(
        repo_root: impl AsRef<Path>,
        cwd: impl AsRef<Path>,
        home: impl AsRef<Path>,
    ) -> Self {
        Layout {
            cwd: cwd.as_ref().to_path_buf(),
            repo_root: repo_root.as_ref().to_path_buf(),
            home: home.as_ref().to_path_buf(),
            managed_settings: None,
            managed_mcp: None,
        }
    }

    pub fn with_managed_settings(mut self, path: impl AsRef<Path>) -> Self {
        self.managed_settings = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn with_managed_mcp(mut self, path: impl AsRef<Path>) -> Self {
        self.managed_mcp = Some(path.as_ref().to_path_buf());
        self
    }

    /// Discover a layout from a starting directory: walk up to the nearest
    /// `.git`, read `$HOME` for the user scope. Falls back to `cwd` as the repo
    /// root when no `.git` is found.
    pub fn discover(cwd: impl AsRef<Path>) -> Self {
        let cwd = cwd.as_ref().to_path_buf();
        let repo_root = find_repo_root(&cwd).unwrap_or_else(|| cwd.clone());
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| cwd.clone());
        Layout {
            cwd,
            repo_root,
            home,
            managed_settings: None,
            managed_mcp: None,
        }
    }

    /// The directories from `repo_root` down to `cwd`, root first. When `cwd` is
    /// not inside `repo_root` the chain collapses to just `cwd`.
    pub fn dir_chain_root_first(&self) -> Vec<PathBuf> {
        let cwd = normalize(&self.cwd);
        let root = normalize(&self.repo_root);

        // Collect cwd and ancestors until we pass root.
        let mut up: Vec<PathBuf> = Vec::new();
        let mut cursor = cwd.clone();
        loop {
            up.push(cursor.clone());
            if cursor == root {
                up.reverse();
                return up;
            }
            match cursor.parent() {
                Some(p) => cursor = p.to_path_buf(),
                None => break,
            }
        }
        // root was never reached (cwd outside repo_root): just cwd.
        vec![cwd]
    }
}

fn normalize(p: &Path) -> PathBuf {
    // Lexical normalisation only (no filesystem access): collapse `.` segments.
    // We do not resolve symlinks; the readers operate on the paths as given.
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let mut cursor = start;
    loop {
        if cursor.join(".git").exists() {
            return Some(cursor.to_path_buf());
        }
        cursor = cursor.parent()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chain_is_root_first_and_inclusive() {
        let layout = Layout::new("/repo", "/repo/a/b", "/home/u");
        let chain = layout.dir_chain_root_first();
        assert_eq!(chain, vec![ PathBuf::from("/repo"), PathBuf::from("/repo/a"), PathBuf::from("/repo/a/b"), ]);
    }
    #[test]
    fn chain_when_cwd_equals_root() {
        let layout = Layout::new("/repo", "/repo", "/home/u");
        assert_eq!(layout.dir_chain_root_first(), vec![PathBuf::from("/repo")]);
    }
    #[test]
    fn chain_when_cwd_outside_root() {
        let layout = Layout::new("/repo", "/elsewhere/x", "/home/u");
 assert_eq!( layout.dir_chain_root_first(), vec![PathBuf::from("/elsewhere/x")] );
    }
}
}


// --- inlined compat/mcp.rs ---
pub mod mcp {
//! MCP server definitions with layered, whole-entry-wins precedence, plus
//! cross-agent config discovery (AGENTS.md and `.cursor/rules`).
//!
//! Servers are read from the project `.mcp.json`, the user `~/.claude.json`, and
//! an optional managed config. For a given server name the highest-precedence
//! scope's WHOLE entry wins (no deep field merge). Precedence is
//! Managed > Project > User.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;

use crate::compat::error::{CompatError, Result};
use crate::compat::layout::Layout;

/// The scope an MCP server entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpScope {
    User,
    Project,
    Managed,
}

/// One MCP server entry, kept as its raw JSON so no fields are lost.
#[derive(Debug, Clone)]
pub struct McpServer {
    pub name: String,
    pub scope: McpScope,
    pub entry: Json,
}

impl McpServer {
    /// Convenience: the `command` field if present.
    pub fn command(&self) -> Option<&str> {
        self.entry.get("command").and_then(|v| v.as_str())
    }

    /// Convenience: the transport `type`/`transport` field if present.
    pub fn transport(&self) -> Option<&str> {
        self.entry
            .get("type")
            .or_else(|| self.entry.get("transport"))
            .and_then(|v| v.as_str())
    }
}

/// The resolved MCP configuration plus adjacent cross-agent config.
#[derive(Debug, Clone, Default)]
pub struct McpConfig {
    /// Servers by name after whole-entry precedence resolution.
    pub servers: BTreeMap<String, McpServer>,
    /// Content of a top-level `AGENTS.md`, if present.
    pub agents_md: Option<String>,
    /// Cursor rules discovered under `.cursor/rules` (and `.cursorrules`).
    pub cursor_rules: Vec<CursorRule>,
}

impl McpConfig {
    pub fn server(&self, name: &str) -> Option<&McpServer> {
        self.servers.get(name)
    }
}

/// A single Cursor rule file.
#[derive(Debug, Clone)]
pub struct CursorRule {
    pub path: PathBuf,
    pub body: String,
}

/// Load and resolve the MCP configuration for a layout.
pub fn load(layout: &Layout) -> Result<McpConfig> {
    // Read low -> high precedence so higher scopes overwrite whole entries.
    let user_servers = read_servers(&layout.home.join(".claude.json"), McpScope::User)?;
    let project_servers = read_servers(&layout.repo_root.join(".mcp.json"), McpScope::Project)?;
    let managed_servers = match &layout.managed_mcp {
        Some(p) => read_servers(p, McpScope::Managed)?,
        None => Vec::new(),
    };

    let mut servers: BTreeMap<String, McpServer> = BTreeMap::new();
    for s in user_servers
        .into_iter()
        .chain(project_servers)
        .chain(managed_servers)
    {
        // Whole-entry-wins: a later (higher precedence) scope replaces the entry.
        servers.insert(s.name.clone(), s);
    }

    let agents_md = read_optional(&layout.repo_root.join("AGENTS.md"));
    let cursor_rules = discover_cursor_rules(layout);

    Ok(McpConfig {
        servers,
        agents_md,
        cursor_rules,
    })
}

fn read_servers(path: &Path, scope: McpScope) -> Result<Vec<McpServer>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(CompatError::Io {
                path: path.display().to_string(),
                source: e,
            })
        }
    };
    let json: Json = serde_json::from_str(&text).map_err(|e| CompatError::Json {
        path: path.display().to_string(),
        source: e,
    })?;
    Ok(extract_servers(&json, scope))
}

/// Pull `mcpServers` (and, for `~/.claude.json`, nested per-project maps) out of
/// a decoded JSON document.
fn extract_servers(json: &Json, scope: McpScope) -> Vec<McpServer> {
    let mut out: Vec<McpServer> = Vec::new();

    if let Some(map) = json.get("mcpServers").and_then(|v| v.as_object()) {
        for (name, entry) in map {
            out.push(McpServer {
                name: name.clone(),
                scope,
                entry: entry.clone(),
            });
        }
    }

    // ~/.claude.json also stores per-project blocks under `projects`; each may
    // carry its own mcpServers. We flatten them at the same (user) scope; a
    // later duplicate name simply overwrites within this scope's vector, which
    // is fine because cross-scope precedence is applied by the caller.
    if let Some(projects) = json.get("projects").and_then(|v| v.as_object()) {
        for block in projects.values() {
            if let Some(map) = block.get("mcpServers").and_then(|v| v.as_object()) {
                for (name, entry) in map {
                    out.push(McpServer {
                        name: name.clone(),
                        scope,
                        entry: entry.clone(),
                    });
                }
            }
        }
    }

    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn read_optional(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn discover_cursor_rules(layout: &Layout) -> Vec<CursorRule> {
    let mut rules: Vec<CursorRule> = Vec::new();

    // Legacy single-file `.cursorrules`.
    let legacy = layout.repo_root.join(".cursorrules");
    if let Some(body) = read_optional(&legacy) {
        rules.push(CursorRule {
            path: legacy,
            body,
        });
    }

    // Modern `.cursor/rules/**` (typically `*.mdc`, but read any file).
    let dir = layout.repo_root.join(".cursor").join("rules");
    if dir.exists() {
        let mut files: Vec<PathBuf> = Vec::new();
        for entry in walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                files.push(entry.into_path());
            }
        }
        files.sort();
        for path in files {
            if let Some(body) = read_optional(&path) {
                rules.push(CursorRule { path, body });
            }
        }
    }

    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn extract_pulls_top_level_and_project_servers() {
        let doc = json!({
            "mcpServers": {"a": {"command": "a-cmd"}},
            "projects": {"/some/repo": {"mcpServers": {"b": {"command": "b-cmd"}}}}
        });
        let servers = extract_servers(&doc, McpScope::User);
        let names: Vec<&str> = servers.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }
    #[test]
    fn whole_entry_wins_no_deep_merge() {
        let user = McpServer {
            name: "srv".into(),
            scope: McpScope::User,
            entry: json!({"command": "user-cmd", "args": ["--user"]}),
        };
        let project = McpServer {
            name: "srv".into(),
            scope: McpScope::Project,
            entry: json!({"command": "project-cmd"}),
        };
        let mut servers: BTreeMap<String, McpServer> = BTreeMap::new();
        servers.insert(user.name.clone(), user);
        servers.insert(project.name.clone(), project);
        let s = servers.get("srv").unwrap();
        assert_eq!(s.command(), Some("project-cmd"));
        assert!(s.entry.get("args").is_none());
    }
}
}


// --- inlined compat/rules.rs ---
pub mod rules {
//! `.claude/rules/**/*.md` discovery and path-glob gating.
//!
//! A rule with no `paths` frontmatter is *un-scoped*: it loads at launch with
//! the same standing as CLAUDE.md memory. A rule that declares `paths` globs is
//! *scoped*: it only applies when a file matching one of those globs is read.

use std::path::{Path, PathBuf};

use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::compat::frontmatter;
use crate::compat::layout::Layout;

/// A discovered rule file.
#[derive(Debug, Clone)]
pub struct Rule {
    pub path: PathBuf,
    /// Raw path globs from frontmatter (empty when un-scoped).
    pub paths: Vec<String>,
    /// Whether the rule is scoped to matching-file reads.
    pub scoped: bool,
    /// The rule body (frontmatter stripped).
    pub body: String,
    /// Compiled glob set for `paths`, if any.
    glob_set: Option<GlobSet>,
}

impl Rule {
    /// Un-scoped rules load at launch alongside CLAUDE.md.
    pub fn loads_at_launch(&self) -> bool {
        !self.scoped
    }

    /// Whether this rule applies when `file` (relative to the repo root, or
    /// absolute) is read. Un-scoped rules always apply.
    pub fn applies_to(&self, file: &Path, repo_root: &Path) -> bool {
        if !self.scoped {
            return true;
        }
        let set = match &self.glob_set {
            Some(s) => s,
            None => return false,
        };
        let rel = file.strip_prefix(repo_root).unwrap_or(file);
        set.is_match(file) || set.is_match(rel)
    }
}

/// Discover rules under `<dir>/.claude/rules/**/*.md` for both the repo root and
/// the user home. Sorted by path for deterministic ordering.
pub fn discover(layout: &Layout) -> Vec<Rule> {
    let mut roots = vec![layout.repo_root.join(".claude").join("rules")];
    roots.push(layout.home.join(".claude").join("rules"));

    let mut files: Vec<PathBuf> = Vec::new();
    for root in roots {
        if !root.exists() {
            continue;
        }
        for entry in walkdir::WalkDir::new(&root)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file()
                && entry.path().extension().map(|e| e == "md").unwrap_or(false)
            {
                files.push(entry.into_path());
            }
        }
    }
    files.sort();
    files.dedup();

    files
        .into_iter()
        .filter_map(|path| parse_rule(&path))
        .collect()
}

/// Parse a single rule file.
pub fn parse_rule(path: &Path) -> Option<Rule> {
    let raw = std::fs::read_to_string(path).ok()?;
    let (fm, body) = frontmatter::split(&raw);
    let paths = fm
        .as_ref()
        .map(|f| {
            let mut p = f.list("paths");
            if p.is_empty() {
                // Some ecosystems use `globs:` for the same concept.
                p = f.list("globs");
            }
            p
        })
        .unwrap_or_default();
    let scoped = !paths.is_empty();
    let glob_set = if scoped {
        build_glob_set(&paths).ok()
    } else {
        None
    };
    Some(Rule {
        path: path.to_path_buf(),
        paths,
        scoped,
        body,
        glob_set,
    })
}

fn build_glob_set(patterns: &[String]) -> Result<GlobSet, globset::Error> {
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        builder.add(Glob::new(p)?);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unscoped_rule_loads_at_launch_and_always_applies() {
        let rule = Rule {
            path: PathBuf::from("/repo/.claude/rules/general.md"),
            paths: vec![],
            scoped: false,
            body: "always".into(),
            glob_set: None,
        };
        assert!(rule.loads_at_launch());
        assert!(rule.applies_to(Path::new("/repo/anything.txt"), Path::new("/repo")));
    }
    #[test]
    fn scoped_rule_gates_on_glob() {
        let set = build_glob_set(&["**/*.rs".to_string()]).unwrap();
        let rule = Rule {
            path: PathBuf::from("/repo/.claude/rules/rust.md"),
            paths: vec!["**/*.rs".into()],
            scoped: true,
            body: "rust only".into(),
            glob_set: Some(set),
        };
        assert!(!rule.loads_at_launch());
        assert!(rule.applies_to(Path::new("/repo/src/main.rs"), Path::new("/repo")));
        assert!(!rule.applies_to(Path::new("/repo/README.md"), Path::new("/repo")));
    }
}
}


// --- inlined compat/settings.rs ---
pub mod settings {
//! settings.json readers across scopes, with two separate precedence rules.
//!
//! Scalar settings resolve with precedence Managed > CLI > Local > Project >
//! User (highest wins). Permission arrays (`allow`/`deny`/`ask`) do not override;
//! they MERGE across every scope, and within the merged set a `deny` beats an
//! `allow` beats an `ask`. Instruction layers use a *separate* precedence,
//! Managed > User > Project > Local, applied read-last-wins (so the highest
//! precedence layer is read last).

use std::collections::BTreeMap;

use globset::{Glob, GlobSet, GlobSetBuilder};
use serde_json::Value as Json;

use crate::compat::error::{CompatError, Result};
use crate::compat::layout::Layout;

/// A configuration scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
    User,
    Project,
    Local,
    Cli,
    Managed,
}

/// The permission decision for a tool invocation string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
    Ask,
    /// No rule matched in any scope.
    Undecided,
}

/// Merged permission rule sets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Permissions {
    pub allow: Vec<String>,
    pub deny: Vec<String>,
    pub ask: Vec<String>,
}

impl Permissions {
    fn extend_from(&mut self, other: &Permissions) {
        for a in &other.allow {
            if !self.allow.contains(a) {
                self.allow.push(a.clone());
            }
        }
        for d in &other.deny {
            if !self.deny.contains(d) {
                self.deny.push(d.clone());
            }
        }
        for k in &other.ask {
            if !self.ask.contains(k) {
                self.ask.push(k.clone());
            }
        }
    }

    /// Decide a tool string. Deny wins, then allow, then ask. Matching is exact
    /// on the rule string OR by a glob compiled from the rule (so a rule like
    /// `Bash(git *)` matches `Bash(git status)`).
    pub fn decide(&self, tool: &str) -> Decision {
        if matches_any(&self.deny, tool) {
            Decision::Deny
        } else if matches_any(&self.allow, tool) {
            Decision::Allow
        } else if matches_any(&self.ask, tool) {
            Decision::Ask
        } else {
            Decision::Undecided
        }
    }
}

fn matches_any(rules: &[String], tool: &str) -> bool {
    for r in rules {
        if r == tool {
            return true;
        }
        // Best-effort glob: only attempt when the rule looks like a pattern.
        if r.contains('*') {
            if let Ok(glob) = Glob::new(r) {
                if glob.compile_matcher().is_match(tool) {
                    return true;
                }
            }
        }
    }
    false
}

/// One scope's raw settings, parsed from a settings.json.
#[derive(Debug, Clone, Default)]
pub struct RawSettings {
    pub scope_present: bool,
    pub permissions: Permissions,
    pub claude_md_excludes: Vec<String>,
    /// The instruction string a scope contributes (from `instructions`, falling
    /// back to `additionalInstructions`).
    pub instructions: Option<String>,
    /// All top-level scalar/object settings, for scalar precedence resolution.
    pub values: BTreeMap<String, Json>,
}

/// The fully resolved settings for a layout.
#[derive(Debug, Clone, Default)]
pub struct ResolvedSettings {
    /// Merged permission rules (deny wins on decision).
    pub permissions: Permissions,
    /// Scalar settings after Managed > CLI > Local > Project > User resolution.
    pub values: BTreeMap<String, Json>,
    /// Instruction layers in application order (read-last wins). The last element
    /// is the highest precedence (Managed if present).
    pub instruction_layers: Vec<(Scope, String)>,
    /// Merged `claudeMdExcludes` globs across all scopes.
    pub claude_md_excludes: Vec<String>,
}

impl ResolvedSettings {
    pub fn decide(&self, tool: &str) -> Decision {
        self.permissions.decide(tool)
    }

    /// Compile `claudeMdExcludes` into a glob set for CLAUDE.md discovery.
    pub fn excludes_glob_set(&self) -> Result<Option<GlobSet>> {
        if self.claude_md_excludes.is_empty() {
            return Ok(None);
        }
        let mut builder = GlobSetBuilder::new();
        for g in &self.claude_md_excludes {
            let glob = Glob::new(g).map_err(|e| CompatError::Glob {
                glob: g.clone(),
                source: e,
            })?;
            builder.add(glob);
        }
        let set = builder.build().map_err(|e| CompatError::Glob {
            glob: self.claude_md_excludes.join(","),
            source: e,
        })?;
        Ok(Some(set))
    }

    /// The effective (highest-precedence) instruction string, or None.
    pub fn effective_instructions(&self) -> Option<&str> {
        self.instruction_layers.last().map(|(_, s)| s.as_str())
    }
}

/// Parse a settings.json file into `RawSettings`. Missing file is not an error;
/// it yields an absent scope.
pub fn parse_file(path: &std::path::Path) -> Result<RawSettings> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(RawSettings::default());
        }
        Err(e) => {
            return Err(CompatError::Io {
                path: path.display().to_string(),
                source: e,
            })
        }
    };
    let json: Json = serde_json::from_str(&text).map_err(|e| CompatError::Json {
        path: path.display().to_string(),
        source: e,
    })?;
    Ok(parse_value(json))
}

/// Parse an already-decoded JSON object into `RawSettings` (used for CLI scope).
pub fn parse_value(json: Json) -> RawSettings {
    let mut raw = RawSettings {
        scope_present: true,
        ..Default::default()
    };
    let obj = match json.as_object() {
        Some(o) => o,
        None => return raw,
    };

    if let Some(perms) = obj.get("permissions").and_then(|v| v.as_object()) {
        raw.permissions.allow = string_array(perms.get("allow"));
        raw.permissions.deny = string_array(perms.get("deny"));
        raw.permissions.ask = string_array(perms.get("ask"));
    }

    raw.claude_md_excludes = string_array(obj.get("claudeMdExcludes"));

    raw.instructions = obj
        .get("instructions")
        .and_then(|v| v.as_str())
        .or_else(|| obj.get("additionalInstructions").and_then(|v| v.as_str()))
        .map(|s| s.to_string());

    for (k, v) in obj {
        raw.values.insert(k.clone(), v.clone());
    }

    raw
}

fn string_array(v: Option<&Json>) -> Vec<String> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Load and resolve settings for a layout. `cli` optionally supplies the CLI
/// scope (from parsed command-line flags).
pub fn load(layout: &Layout, cli: Option<RawSettings>) -> Result<ResolvedSettings> {
    let user = parse_file(&layout.home.join(".claude").join("settings.json"))?;
    let project = parse_file(&layout.repo_root.join(".claude").join("settings.json"))?;
    let local = parse_file(
        &layout
            .repo_root
            .join(".claude")
            .join("settings.local.json"),
    )?;
    let managed = match &layout.managed_settings {
        Some(p) => parse_file(p)?,
        None => RawSettings::default(),
    };
    let cli = cli.unwrap_or_default();

    resolve(&user, &project, &local, &cli, &managed)
}

/// Resolve raw scopes into a `ResolvedSettings` (exposed for direct testing).
pub fn resolve(
    user: &RawSettings,
    project: &RawSettings,
    local: &RawSettings,
    cli: &RawSettings,
    managed: &RawSettings,
) -> Result<ResolvedSettings> {
    // Permissions MERGE across every present scope; order does not matter since
    // deny/allow/ask are unioned and decided by deny-wins.
    let mut permissions = Permissions::default();
    let mut claude_md_excludes: Vec<String> = Vec::new();
    for s in [user, project, local, cli, managed] {
        if !s.scope_present {
            continue;
        }
        permissions.extend_from(&s.permissions);
        for g in &s.claude_md_excludes {
            if !claude_md_excludes.contains(g) {
                claude_md_excludes.push(g.clone());
            }
        }
    }

    // Scalar values: apply low -> high so the highest precedence overwrites.
    // Precedence: Managed > CLI > Local > Project > User.
    let mut values: BTreeMap<String, Json> = BTreeMap::new();
    for s in [user, project, local, cli, managed] {
        if !s.scope_present {
            continue;
        }
        for (k, v) in &s.values {
            values.insert(k.clone(), v.clone());
        }
    }

    // Instruction layers: SEPARATE precedence Managed > User > Project > Local.
    // read-last-wins => apply lowest first: Local, Project, User, Managed.
    let mut instruction_layers: Vec<(Scope, String)> = Vec::new();
    let ordered = [
        (Scope::Local, local),
        (Scope::Project, project),
        (Scope::User, user),
        (Scope::Managed, managed),
    ];
    for (scope, s) in ordered {
        if !s.scope_present {
            continue;
        }
        if let Some(instr) = &s.instructions {
            instruction_layers.push((scope, instr.clone()));
        }
    }

    Ok(ResolvedSettings {
        permissions,
        values,
        instruction_layers,
        claude_md_excludes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn raw(json: Json) -> RawSettings {
        parse_value(json)
    }
    #[test]
    fn permissions_merge_and_deny_wins() {
        let user = raw(json!({"permissions": {"allow": ["Bash(git status)"]}}));
        let project = raw(json!({"permissions": {"deny": ["Bash(git status)"]}}));
        let empty = RawSettings::default();
        let resolved = resolve(&user, &project, &empty, &empty, &empty).unwrap();
        assert_eq!(resolved.decide("Bash(git status)"), Decision::Deny);
    }
    #[test]
    fn scalar_precedence_managed_over_user() {
        let user = raw(json!({"model": "user-model"}));
        let managed = raw(json!({"model": "managed-model"}));
        let empty = RawSettings::default();
        let resolved = resolve(&user, &empty, &empty, &empty, &managed).unwrap();
 assert_eq!( resolved.values.get("model").and_then(|v| v.as_str()), Some("managed-model") );
    }
    #[test]
    fn instruction_layers_read_last_wins_managed_highest() {
        let local = raw(json!({"instructions": "local"}));
        let project = raw(json!({"instructions": "project"}));
        let user = raw(json!({"instructions": "user"}));
        let managed = raw(json!({"instructions": "managed"}));
        let resolved = resolve(&user, &project, &local, &RawSettings::default(), &managed).unwrap();
        let order: Vec<Scope> = resolved.instruction_layers.iter().map(|(s, _)| *s).collect();
 assert_eq!( order, vec![Scope::Local, Scope::Project, Scope::User, Scope::Managed] );
        assert_eq!(resolved.effective_instructions(), Some("managed"));
    }
    #[test]
    fn glob_permission_matches() {
        let project = raw(json!({"permissions": {"allow": ["Bash(git *)"]}}));
        let empty = RawSettings::default();
        let resolved = resolve(&empty, &project, &empty, &empty, &empty).unwrap();
        assert_eq!(resolved.decide("Bash(git push)"), Decision::Allow);
    }
}
}


// --- inlined compat/skills.rs ---
pub mod skills {
//! Skill definitions: `SKILL.md` frontmatter.
//!
//! Skills live under `<scope>/.claude/skills/<name>/SKILL.md`. The frontmatter
//! carries invocation metadata (`allowed-tools`, `disable-model-invocation`,
//! `user-invocable`, `context`, `model`, `effort`, `paths`).

use std::path::{Path, PathBuf};

use crate::compat::frontmatter::{self, Frontmatter};
use crate::compat::layout::Layout;

/// A parsed skill.
#[derive(Debug, Clone)]
pub struct Skill {
    pub path: PathBuf,
    pub name: String,
    pub description: Option<String>,
    pub allowed_tools: Vec<String>,
    /// When true the model may not auto-invoke the skill (user must trigger it).
    pub disable_model_invocation: bool,
    /// When true the user can invoke the skill directly (slash command).
    pub user_invocable: bool,
    pub context: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
    /// Path globs the skill is scoped to (empty = always available).
    pub paths: Vec<String>,
    /// The skill body (frontmatter stripped).
    pub body: String,
}

impl Skill {
    /// Whether the model is allowed to auto-invoke this skill.
    pub fn model_invocable(&self) -> bool {
        !self.disable_model_invocation
    }
}

/// Parse a single SKILL.md file.
pub fn parse(path: &Path) -> Option<Skill> {
    let raw = std::fs::read_to_string(path).ok()?;
    let (fm, body) = frontmatter::split(&raw);
    let fm = fm.unwrap_or_default();
    Some(from_frontmatter(path, &fm, body))
}

fn from_frontmatter(path: &Path, fm: &Frontmatter, body: String) -> Skill {
    let name = fm.str("name").unwrap_or_else(|| {
        // Fall back to the enclosing directory name (skills/<name>/SKILL.md).
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("skill")
            .to_string()
    });
    Skill {
        path: path.to_path_buf(),
        name,
        description: fm.str("description"),
        allowed_tools: fm.list("allowed-tools"),
        // Default false: model may invoke unless explicitly disabled.
        disable_model_invocation: fm.bool("disable-model-invocation").unwrap_or(false),
        // Default true: user-invocable unless explicitly disabled.
        user_invocable: fm.bool("user-invocable").unwrap_or(true),
        context: fm.str("context"),
        model: fm.str("model"),
        effort: fm.str("effort"),
        paths: fm.list("paths"),
        body,
    }
}

/// Discover skills under the project and user `.claude/skills` trees. Project
/// skills win by name. Sorted by name for determinism.
pub fn discover(layout: &Layout) -> Vec<Skill> {
    let mut by_name: std::collections::BTreeMap<String, Skill> =
        std::collections::BTreeMap::new();

    for dir in [
        layout.home.join(".claude").join("skills"),
        layout.repo_root.join(".claude").join("skills"),
    ] {
        for skill in parse_dir(&dir) {
            by_name.insert(skill.name.clone(), skill);
        }
    }

    by_name.into_values().collect()
}

fn parse_dir(dir: &Path) -> Vec<Skill> {
    if !dir.exists() {
        return Vec::new();
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() && entry.file_name() == "SKILL.md" {
            files.push(entry.into_path());
        }
    }
    files.sort();
    files.iter().filter_map(|p| parse(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_skill_flags_with_defaults() {
        let fm = frontmatter::parse_block(
            "name: deploy\ndescription: ship it\nallowed-tools: [Bash, Read]\ndisable-model-invocation: true\neffort: high\n",
        );
        let skill = from_frontmatter(Path::new("/x/deploy/SKILL.md"), &fm, String::new());
        assert_eq!(skill.name, "deploy");
        assert_eq!(skill.allowed_tools, vec!["Bash", "Read"]);
        assert!(!skill.model_invocable());
        assert!(skill.user_invocable);
        assert_eq!(skill.effort.as_deref(), Some("high"));
    }
    #[test]
    fn name_falls_back_to_dir() {
        let fm = frontmatter::parse_block("description: no name here\n");
        let skill = from_frontmatter(Path::new("/x/myskill/SKILL.md"), &fm, String::new());
        assert_eq!(skill.name, "myskill");
    }
}
}

