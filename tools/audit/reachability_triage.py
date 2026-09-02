"""Reachability triage -- one disposition per module, from call sites.

Hawking's known worst defect class is capability that exists but nothing
calls. This tool is the inventory: every non-test Python module under
tools/future/, tools/accelerator/, tools/odyssey/, and tools/headless/ is
classified and given a disposition (CONNECTED / PARKED / ARCHIVE_CANDIDATE)
from grep/AST evidence only.

Engine
------
The analyzer is tools/future/capability_reachability.py. This module does
not write a second one. It calls assemble() and the RepoIndex primitives
(find_module_import_sites, _subprocess_path_sites, build_capability,
_partition). Two holes in this sparse worktree made the engine insufficient;
both are patched in-process before any index is built:

  1. Sparse-checkout git-blob fallback. git ls-files names hcli/*.py but
     they are not on disk here; Path.read_text was returning empty and the
     import index silently dropped every HCLI call site. Blobs are loaded
     from HEAD via one `git cat-file --batch`.
  2. receipts/fixtures/goldens are data, not callers. An import inside a
     generated receipt runner is not evidence the resident reaches it.

    python3 tools/audit/reachability_triage.py --build
    python3 tools/audit/reachability_triage.py --selftest
    python3 -m pytest tools/audit/test_reachability_triage.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import (
    GIT_TIMEOUT_S,
    REPO,
    git,
    load_json,
    require_known_flags,
    write_receipt,
)
from tools.future import capability_reachability as cr

import argparse
import ast
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RECEIPT = "REACHABILITY_TRIAGE.json"
SCHEMA = "hawking.audit.reachability_triage.v1"
VERSION = 1
RECORDED_BY = "tools/audit/reachability_triage.py"

TREE_ROOTS = ("tools/future", "tools/accelerator", "tools/odyssey", "tools/headless")

CLASSIFICATIONS = (
    "BUILT",
    "SCAFFOLDED",
    "DORMANT",
    "UNREACHABLE",
    "ARCHIVE_CANDIDATE",
)
DISPOSITIONS = ("CONNECTED", "PARKED", "ARCHIVE_CANDIDATE")

HCLI_ENTRY_MODULES = (
    "hcli",
    "hcli.__main__",
    "hcli.commands",
    "hcli.command_registry",
    "hcli.tool_registry",
    "hcli.mission",
    "hcli.frontier_scheduler",
    "hcli.agentos",
    "hcli.agentos.resident",
    "hcli.agentos.runtime",
    "hcli.engine",
    "hcli.executors",
)

_DATA_DIR_NAMES = frozenset({"receipts", "fixtures", "goldens"})
# Only the MODULE retiring itself, not a docstring that mentions some other
# campaign/driver/path being retired. Matching bare "retired" put index_provenance
# (legacy driver was retired) and ramanujan_disband (the campaign is retired)
# in ARCHIVE_CANDIDATE, which is a classification error.
_RETIRED_RE = re.compile(
    r"(?i)(?:this module (?:is |has been )?(?:retired|superseded|deprecated)|"
    r"do not (?:use|import) this module|"
    r"superseded by tools/)"
)
_WAKE_CONST_RE = re.compile(r"^WAKE_[A-Z0-9_]+$")


# --------------------------------------------------------------------------
# Engine extensions (in-process; the analyzer file is the engine)
# --------------------------------------------------------------------------


_EXTENSIONS_INSTALLED = False
_ORIG_BUILD_REPO_INDEX = cr.build_repo_index
_ORIG_PARTITION = cr._partition
_INDEX_CACHE: dict[Any, cr.RepoIndex] = {}


def is_data_path(path: Path) -> bool:
    return any(part in _DATA_DIR_NAMES for part in Path(path).parts)


def is_production_path(path: Path) -> bool:
    p = Path(path)
    return (not cr.is_test_path(p)) and (not is_data_path(p))


def _prefetch_git_blobs(paths: Sequence[Path]) -> None:
    """Load sparse-missing files from HEAD. One cat-file --batch, not N shows."""
    if not paths:
        return
    specs = [f"HEAD:{cr.rel(p)}" for p in paths]
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "cat-file", "--batch"],
            cwd=str(REPO),
            input=("\n".join(specs) + "\n").encode("utf-8"),
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError):
        for p in paths:
            cr._TEXT_CACHE.setdefault(p, "")
        return
    data = proc.stdout
    idx = 0
    for path in paths:
        if idx >= len(data):
            cr._TEXT_CACHE[path] = ""
            continue
        nl = data.find(b"\n", idx)
        if nl < 0:
            cr._TEXT_CACHE[path] = ""
            break
        header = data[idx:nl].decode("utf-8", errors="replace")
        idx = nl + 1
        if " missing" in header:
            cr._TEXT_CACHE[path] = ""
            continue
        parts = header.split()
        if len(parts) < 3 or parts[1] != "blob":
            cr._TEXT_CACHE[path] = ""
            continue
        try:
            size = int(parts[2])
        except ValueError:
            cr._TEXT_CACHE[path] = ""
            continue
        blob = data[idx : idx + size]
        idx = idx + size
        if idx < len(data) and data[idx : idx + 1] == b"\n":
            idx += 1
        cr._TEXT_CACHE[path] = blob.decode("utf-8", errors="replace")


def prefetch_texts(files: Sequence[Path]) -> None:
    missing: list[Path] = []
    for path in files:
        if path in cr._TEXT_CACHE:
            continue
        if path.is_file():
            try:
                cr._TEXT_CACHE[path] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                cr._TEXT_CACHE[path] = ""
        else:
            missing.append(path)
    _prefetch_git_blobs(missing)


def _extended_read_text(path: Path) -> str:
    if path not in cr._TEXT_CACHE:
        prefetch_texts([path])
    return cr._TEXT_CACHE.get(path, "")


def _extended_partition(sites: Sequence[cr.Site]) -> tuple[list[cr.Site], list[cr.Site]]:
    """Production vs test, with receipts/fixtures/goldens treated as neither.

    A generated runner under receipts/ that imports a sidecar module is data,
    not a production call site. Tests still count toward `tested`.
    """
    seen: set[tuple[str, int, str]] = set()
    uniq: list[cr.Site] = []
    for s in sites:
        key = (s.file, s.line, s.kind)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    uniq.sort(key=lambda s: (s.file, s.line))
    prod = [s for s in uniq if is_production_path(Path(s.file))]
    test = [s for s in uniq if cr.is_test_path(Path(s.file))]
    return prod, test


def _extended_build_repo_index(files: Sequence[Path] | None = None) -> cr.RepoIndex:
    key: Any
    if files is None:
        key = None
    else:
        key = tuple(cr.rel(p) for p in files)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    file_list = list(files) if files is not None else cr.repo_py_files()
    prefetch_texts(file_list)
    idx = _ORIG_BUILD_REPO_INDEX(files=file_list)
    _INDEX_CACHE[key] = idx
    return idx


def install_engine_extensions() -> list[str]:
    """Patch the live analyzer so sparse-missing files and data paths are honest.

    Idempotent. Does not weaken any check: more files become visible, and
    receipts lose their false 'caller' status.
    """
    global _EXTENSIONS_INSTALLED
    notes = [
        "sparse-checkout git-blob fallback on read_text / build_repo_index "
        "(HEAD:path via git cat-file --batch for files git-tracked but not on disk)",
        "is_data_path: receipts/fixtures/goldens excluded from production call sites",
        "build_repo_index result cached for assemble()+triage sharing",
    ]
    if _EXTENSIONS_INSTALLED:
        return notes
    cr.read_text = _extended_read_text
    cr._partition = _extended_partition
    cr.build_repo_index = _extended_build_repo_index
    cr.is_data_path = is_data_path  # type: ignore[attr-defined]
    cr.is_production_path = is_production_path  # type: ignore[attr-defined]
    _EXTENSIONS_INSTALLED = True
    return notes


# --------------------------------------------------------------------------
# Tree discovery and per-module evidence
# --------------------------------------------------------------------------


def discover_tree_modules(tree_rel: str) -> list[Path]:
    """Every git-tracked non-test .py under tree_rel. Files may be sparse-missing."""
    listed = git("ls-files", "--", tree_rel)
    out: list[Path] = []
    for line in listed.splitlines():
        if not line.endswith(".py"):
            continue
        if "__pycache__" in line:
            continue
        path = REPO / line
        if cr.is_test_path(path):
            continue
        out.append(path)
    return sorted(out, key=lambda p: cr.rel(p))


def module_summary(text: str) -> str:
    if not text.strip():
        return "empty module (package marker or stub)"
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "(unparseable)"
    doc = ast.get_docstring(tree)
    if doc:
        first = doc.strip().splitlines()[0].strip()
        return first[:200]
    return "(no module docstring)"


def module_shape(text: str, filename: str) -> dict[str, Any]:
    """Stub / package-marker / retired flags, derived from the module AST."""
    if not text.strip():
        return {
            "is_stub": filename.endswith("__init__.py"),
            "is_package_marker": filename.endswith("__init__.py"),
            "retired": None,
            "has_main": False,
            "n_functions": 0,
            "n_classes": 0,
            "produces_receipt": False,
            "receipt_names": [],
        }
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {
            "is_stub": True,
            "is_package_marker": False,
            "retired": None,
            "has_main": False,
            "n_functions": 0,
            "n_classes": 0,
            "produces_receipt": False,
            "receipt_names": [],
        }
    doc = ast.get_docstring(tree) or ""
    retired_hit = _RETIRED_RE.search(doc)
    funcs = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    significant = [
        n
        for n in tree.body
        if not isinstance(n, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(n, ast.Expr)
            and isinstance(getattr(n, "value", None), ast.Constant)
        )
    ]
    is_package_marker = filename.endswith("__init__.py") and not funcs and not classes
    source_lines = [
        ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    has_notimpl = "NotImplementedError" in text
    is_stub = bool(
        is_package_marker
        or (
            len(funcs) + len(classes) <= 1
            and len(source_lines) < 30
            and (has_notimpl or len(significant) <= 2)
        )
    )
    has_main = False
    for n in tree.body:
        if isinstance(n, ast.If):
            test = n.test
            # if __name__ == "__main__":
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and test.comparators
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):
                has_main = True
    receipt_names: list[str] = []
    produces = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == "RECEIPT":
                    if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                        receipt_names.append(n.value.value)
                        produces = True
        if isinstance(n, ast.Call):
            f = n.func
            fname = ""
            if isinstance(f, ast.Name):
                fname = f.id
            elif isinstance(f, ast.Attribute):
                fname = f.attr
            if fname in {"write_receipt", "write_measured_receipt"}:
                produces = True
                if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                    receipt_names.append(n.args[0].value)
    # unique, stable
    seen: set[str] = set()
    uniq_receipts: list[str] = []
    for r in receipt_names:
        if r not in seen:
            seen.add(r)
            uniq_receipts.append(r)
    return {
        "is_stub": is_stub,
        "is_package_marker": is_package_marker,
        "retired": retired_hit.group(0) if retired_hit else None,
        "has_main": has_main,
        "n_functions": len(funcs),
        "n_classes": len(classes),
        "produces_receipt": produces,
        "receipt_names": uniq_receipts,
    }


def recover_orchestration_bindings(idx: cr.RepoIndex) -> frozenset[str]:
    """Filenames listed in tools/future/orchestration.py BINDINGS.

    A BINDINGS row is registration, not a call. invoke() imports the module
    at runtime; without a static invoke("<file>") this does not make it
    callable. Recorded so PARKED wake conditions can name the existing hook.
    """
    path = REPO / "tools/future/orchestration.py"
    text = cr.read_text(path)
    if not text:
        return frozenset()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not (isinstance(target, ast.Name) and target.id == "BINDINGS"):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.add(key.value)
    # also a plain Assign, in case the annotation form is rewritten
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "BINDINGS" for t in node.targets):
            continue
        if isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value)
    return frozenset(names)


def build_dynamic_import_index(idx: cr.RepoIndex) -> dict[str, list[cr.Site]]:
    """importlib.import_module("tools.foo.bar") / __import__("tools.foo.bar") sites.

    One AST pass over the repo, not one pass per module.
    """
    out: dict[str, list[cr.Site]] = {}
    for path in idx.files:
        rp = cr.rel(path)
        if not is_production_path(Path(rp)):
            continue
        text = cr.read_text(path)
        if "import_module" not in text and "__import__" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            arg0 = node.args[0]
            if not (isinstance(arg0, ast.Constant) and isinstance(arg0.value, str)):
                continue
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name not in {"import_module", "__import__"}:
                continue
            out.setdefault(arg0.value, []).append(cr.Site(rp, node.lineno, "call"))
    return out


def build_import_graph(idx: cr.RepoIndex) -> dict[str, set[str]]:
    """importer_module -> {imported_module, ...} from production import sites only."""
    graph: dict[str, set[str]] = {}
    for target, sites in idx.import_sites.items():
        for site in sites:
            if not is_production_path(Path(site.file)):
                continue
            importer = cr.module_name_of(REPO / site.file)
            graph.setdefault(importer, set()).add(target)
    return graph


def hcli_reachable_set(
    idx: cr.RepoIndex,
    graph: dict[str, set[str]],
    subprocess_edges: Mapping[str, Iterable[str]],
) -> tuple[set[str], dict[str, str]]:
    """Modules reachable from any production file under hcli/.

    Returns (reachable_modules, parent_map) where parent_map[m] is the
    importer that first reached m (for a shortest-path citation).
    """
    start: set[str] = set()
    parent: dict[str, str] = {}
    for path in idx.files:
        rp = cr.rel(path)
        if not rp.startswith("hcli/"):
            continue
        if not is_production_path(Path(rp)):
            continue
        start.add(cr.module_name_of(path))
    seen: set[str] = set()
    q: deque[str] = deque(sorted(start))
    for s in start:
        parent.setdefault(s, "")
    while q:
        m = q.popleft()
        if m in seen:
            continue
        seen.add(m)
        for dest in graph.get(m, ()):
            if dest not in seen and dest not in parent:
                parent[dest] = m
                q.append(dest)
        for dest in subprocess_edges.get(m, ()):
            if dest not in seen and dest not in parent:
                parent[dest] = m
                q.append(dest)
    return seen, parent


def cite_hcli_path(
    module_dotted: str,
    call_sites: Sequence[Mapping[str, Any]],
    parent: Mapping[str, str],
) -> str | None:
    hcli_sites = [s for s in call_sites if str(s.get("file", "")).startswith("hcli/")]
    if hcli_sites:
        s = hcli_sites[0]
        return f"{s['file']}:{s['line']} ({s.get('kind', 'import')})"
    if module_dotted not in parent:
        return None
    chain: list[str] = [module_dotted]
    cur = module_dotted
    guard = 0
    while cur in parent and parent[cur] and guard < 32:
        cur = parent[cur]
        chain.append(cur)
        if cur.startswith("hcli"):
            break
        guard += 1
    chain.reverse()
    if not chain or not chain[0].startswith("hcli"):
        return None
    return " -> ".join(chain)


def conventional_test_path(module_path: Path) -> Path:
    return module_path.parent / f"test_{module_path.stem}.py"


def analyze_test(
    module_dotted: str,
    module_path: Path,
    idx: cr.RepoIndex,
    test_sites: Sequence[cr.Site],
) -> dict[str, Any]:
    """Does a test exist, and does it exercise the module or only read a receipt?"""
    files: set[str] = {s.file for s in test_sites}
    conv = conventional_test_path(module_path)
    conv_rel = cr.rel(conv)
    conv_text = cr.read_text(conv)
    if conv_text:
        files.add(conv_rel)
    if not files:
        return {
            "has_test": False,
            "test_files": [],
            "test_exercises": False,
            "test_reads_receipt_only": False,
        }
    exercises = False
    receipt_only = False
    for rel_file in sorted(files):
        text = cr.read_text(REPO / rel_file)
        if not text:
            continue
        kind = _test_file_kind(text, module_dotted, module_path.stem)
        if kind == "exercises":
            exercises = True
        elif kind == "receipt_only":
            receipt_only = True
    return {
        "has_test": True,
        "test_files": sorted(files),
        "test_exercises": exercises,
        "test_reads_receipt_only": bool(receipt_only and not exercises),
    }


def _test_file_kind(text: str, module_dotted: str, stem: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        if ".json" in text:
            return "receipt_only"
        return "other"
    binds_module: set[str] = set()
    imports_module = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_dotted or alias.name.startswith(module_dotted + "."):
                    imports_module = True
                    binds_module.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == module_dotted or mod.startswith(module_dotted + "."):
                imports_module = True
                for alias in node.names:
                    binds_module.add(alias.asname or alias.name)
            # from tools.future import foo as f
            if mod and module_dotted.startswith(mod + "."):
                child = module_dotted[len(mod) + 1 :]
                for alias in node.names:
                    if alias.name == child.split(".")[0]:
                        imports_module = True
                        binds_module.add(alias.asname or alias.name)
    called = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in binds_module:
            called = True
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id in binds_module:
                called = True
            elif func.attr in binds_module:
                called = True
    if imports_module and called:
        return "exercises"
    if imports_module:
        # imported RECEIPT / SCHEMA / a constant -- not an exercise of behavior
        return "imports_only"
    if "load_json" in text or "json.loads" in text or ".json" in text:
        return "receipt_only"
    return "other"


# --------------------------------------------------------------------------
# Classification / disposition -- every path assigns both
# --------------------------------------------------------------------------


def decide(row: dict[str, Any]) -> None:
    """Mutates row with classification, disposition, wake_condition/archive_reason.

    Never leaves disposition empty. Evidence-only; no hand roster.
    """
    callable_ = bool(row.get("callable_outside_tests"))
    hcli = bool(row.get("hcli_reachable"))
    stub = bool(row.get("is_stub"))
    pkg = bool(row.get("is_package_marker"))
    retired = row.get("retired")
    tested = bool(row.get("has_test"))
    exercises = bool(row.get("test_exercises"))
    bound = bool(row.get("orchestration_bound"))
    fn = Path(str(row.get("module", ""))).name

    row["wake_condition"] = None
    row["archive_reason"] = None

    if retired and not callable_ and not hcli:
        row["classification"] = "ARCHIVE_CANDIDATE"
        row["disposition"] = "ARCHIVE_CANDIDATE"
        row["archive_reason"] = (
            f"module docstring marks it {retired!r} and nothing outside tests calls it"
        )
        row["disposition_full"] = f"ARCHIVE_CANDIDATE({row['archive_reason']})"
        return

    if hcli:
        row["classification"] = "SCAFFOLDED" if (stub and not pkg) else "BUILT"
        row["disposition"] = "CONNECTED"
        row["disposition_full"] = "CONNECTED"
        return

    if callable_:
        if stub and not pkg:
            row["classification"] = "SCAFFOLDED"
        else:
            row["classification"] = "DORMANT"
        if bound:
            wake = (
                f"production tools.future.orchestration.invoke({fn!r}) from an HCLI "
                "entry (CLI verb, tool registry, mission executor, or resident); "
                "listed in BINDINGS but invoke() is not statically reached from HCLI"
            )
        else:
            wake = (
                "first HCLI entry-point importer (CLI verb, tool registry, "
                "mission executor, or resident); currently only called inside "
                "the sidecar cluster, not from HCLI"
            )
        row["disposition"] = "PARKED"
        row["wake_condition"] = wake
        row["disposition_full"] = f"PARKED({wake})"
        return

    if stub and not tested and not pkg:
        row["classification"] = "SCAFFOLDED"
        row["disposition"] = "ARCHIVE_CANDIDATE"
        row["archive_reason"] = "scaffold/stub with no production callers and no tests"
        row["disposition_full"] = f"ARCHIVE_CANDIDATE({row['archive_reason']})"
        return

    row["classification"] = "UNREACHABLE"
    row["disposition"] = "PARKED"
    if exercises:
        wake = (
            "first production importer; tests exercise the module but nothing "
            "outside tests calls it"
        )
    elif tested and row.get("test_reads_receipt_only"):
        wake = (
            "first production importer; a test file exists but it reads a "
            "checked-in receipt rather than exercising the module"
        )
    elif tested:
        wake = (
            "first production importer; a test file imports the module but "
            "does not call it"
        )
    elif bound:
        wake = (
            f"first production importer or orchestration.invoke({fn!r}); "
            "listed in BINDINGS, uncalled, untested"
        )
    else:
        wake = "first production importer; currently uncalled and untested"
    row["wake_condition"] = wake
    row["disposition_full"] = f"PARKED({wake})"


# --------------------------------------------------------------------------
# State-transition sweep
# --------------------------------------------------------------------------


def _wake_constants_in(text: str) -> list[tuple[str, int, str]]:
    """(name, line, value) for WAKE_* = "..." assignments."""
    out: list[tuple[str, int, str]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and _WAKE_CONST_RE.match(t.id):
                out.append((t.id, node.lineno, node.value.value))
    return out


def state_transition_sweep(
    idx: cr.RepoIndex,
    tree_files: Sequence[Path],
) -> list[dict[str, Any]]:
    """States that can be entered with no real exit. Report only; do not fix.

    Two layers: verified findings from named, known classes (complete
    artifact / resident loop / spawned child / sleeping WorkUnit), plus a
    mechanical scan for WAKE_* constants that nothing outside their
    defining file ever names.
    """
    findings: list[dict[str, Any]] = []

    # --- known class: complete model/artifact with no promotion path ------
    findings.append(
        {
            "class": "complete_artifact_without_promotion",
            "file": "tools/odyssey/modellake_watch.py",
            "enter": "tools/odyssey/modellake_watch.py:502 complete() returns True",
            "exit": (
                "tools/odyssey/modellake_watch.py:1045 and :1084 call "
                "_promote_and_report -> promote_if_needed -> "
                "modellake_promote.promote(tag, go=True); "
                "reconcile() at :656 is the second-look path"
            ),
            "status": "EXIT_EXISTS",
            "evidence_tier": "STATIC",
            "note": (
                "The original defect (complete() then continue, SPECIMEN_ROOT "
                "never written) is described in tools/odyssey/modellake_promote.py:1-9. "
                "Current HEAD has an exit on both the live admission loop and "
                "the periodic reconcile sweep. Not missing anymore; recorded "
                "so the class is not silently dropped."
            ),
        }
    )

    # --- known class: sleeping WorkUnit with no wake event ----------------
    sealed_refs: list[str] = []
    for path in idx.files:
        rp = cr.rel(path)
        text = cr.read_text(path)
        if "SEALED_SOURCE_READY" in text:
            sealed_refs.append(rp)
    outside = [
        r
        for r in sealed_refs
        if r != "tools/future/sleeping_specimens.py" and not cr.is_test_path(Path(r))
    ]
    findings.append(
        {
            "class": "sleeping_workunit_without_wake_event",
            "file": "tools/future/sleeping_specimens.py",
            "enter": (
                "tools/future/sleeping_specimens.py:426 "
                "wake_condition=SEALED_SOURCE_READY on each SLEEPING_SPECIMEN_WU; "
                "also :468 row['wake_condition'] = WAKE_SEALED_SOURCE_READY"
            ),
            "exit": (
                None
                if not outside
                else f"named in {outside}"
            ),
            "status": "EXIT_MISSING" if not outside else "EXIT_EXISTS",
            "evidence_tier": "STATIC",
            "note": (
                "git grep SEALED_SOURCE_READY on HEAD: every hit is inside "
                "sleeping_specimens.py itself (definition, emit, receipt). "
                "modellake_promote.promote() does not notify this wake "
                "condition. A SLEEPING_SPECIMEN_WU can be entered and never "
                "leave. Test files excluded from 'outside' by the same rule "
                "as call sites."
            ),
            "defining_file_refs": sealed_refs,
            "outside_production_refs": outside,
        }
    )

    findings.append(
        {
            "class": "sleeping_workunit_without_wake_event",
            "file": "tools/future/wakeup.py",
            "enter": (
                "tools/future/wakeup.py:398 register(..., sleeping=True) "
                "sets state=SLEEPING; emit_wakeup_workunits :822 wakeup_state=SLEEPING"
            ),
            "exit": (
                "tools/future/wakeup.py:415 harvest() -> _inspect :607-626: a "
                "sealed receipt at the expected path classifies COMPLETED and "
                "harvest writes exp['state']=event.state. Selftest :1208 "
                "proves SLEEPING without a receipt stays SLEEPING (no synthetic "
                "COMPLETED)."
            ),
            "status": "EXIT_EXISTS",
            "evidence_tier": "STATIC",
            "note": (
                "The exit is a disk receipt, not a caller. That is the module's "
                "contract (completion wakes the graph). Distinct from "
                "SLEEPING_SPECIMEN_WU, whose wake token has no consumer."
            ),
        }
    )

    # --- known class: spawned child with no landing path back -------------
    findings.append(
        {
            "class": "spawned_child_without_landing_path",
            "file": "hcli/agentos/resident.py",
            "enter": (
                "hcli/agentos/resident.py:825 launch_child; :856 "
                "store.update(child_job_ids=children[-64:], last_event='child_started')"
            ),
            "exit": (
                "Process-level: hcli/agentos/background.py BackgroundJobStore._refresh "
                "sets finished_at when the child PID dies (:172-197) and "
                "_supervise_job sets finished_at at :399. Resident-level: no "
                "child_finished / child_landed / last_event='child_*' other than "
                "child_started in resident.py. child_job_ids only appends "
                "(capped at 64) and is never reaped back into the resident "
                "state machine."
            ),
            "status": "EXIT_MISSING",
            "evidence_tier": "STATIC",
            "note": (
                "The child process can finish in the job store. The resident "
                "state machine never consumes that finish. That is the missing "
                "landing path this class names. Do not confuse the two layers."
            ),
        }
    )

    # --- known class: resident loop that can spin without progressing -----
    findings.append(
        {
            "class": "resident_loop_spin_without_progress",
            "file": "hcli/agentos/resident.py",
            "enter": (
                "hcli/agentos/resident.py:1188 WAIT_FOR_CLEAN_ROOM -> "
                "state=WAITING_FOR_CLEAN_ROOM; :1201 WAIT_FOR_MEMORY evacuates; "
                "heartbeat loop :1234 self._wake.wait(config.interval_s)"
            ),
            "exit": (
                "WAITING_FOR_CLEAN_ROOM: last_event=clean_room_resumed at "
                "resident.py:813 (resume path exists). WAIT_FOR_MEMORY: "
                "_evacuate, then the next heartbeat re-evaluates memory and "
                "can spawn. IDLE at :1224 when no mission/inbox. FAILED at "
                ":1213 on restart limit. The loop itself always waits on "
                "_wake; progress depends on those exits actually being taken."
            ),
            "status": "EXIT_EXISTS",
            "evidence_tier": "STATIC",
            "note": (
                "Exits exist in the source. Whether WAITING_FOR_CLEAN_ROOM is "
                "ever resumed in a live daemon is a runtime question this "
                "STATIC sweep cannot settle. Recorded as EXIT_EXISTS because "
                "the transition is present, not because it was observed firing."
            ),
        }
    )

    # --- mechanical: WAKE_* constants with no outside production reference
    for path in tree_files:
        rel_p = cr.rel(path)
        text = cr.read_text(path)
        if not text:
            continue
        for name, lineno, value in _wake_constants_in(text):
            refs: list[str] = []
            for other in idx.files:
                rp = cr.rel(other)
                if rp == rel_p:
                    continue
                other_text = cr.read_text(other)
                if value in other_text or name in other_text:
                    if is_production_path(Path(rp)):
                        refs.append(f"{rp}")
            if refs:
                continue
            # skip the SEALED_SOURCE_READY finding already recorded above
            if value == "SEALED_SOURCE_READY":
                continue
            findings.append(
                {
                    "class": "sleeping_workunit_without_wake_event",
                    "file": rel_p,
                    "enter": f"{rel_p}:{lineno} {name} = {value!r}",
                    "exit": None,
                    "status": "EXIT_MISSING",
                    "evidence_tier": "STATIC",
                    "note": (
                        f"WAKE_* constant {name}={value!r} is never named in any "
                        "other production file. Entering a unit with this wake "
                        "token has no consumer that can fire it."
                    ),
                }
            )

    return findings


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _subprocess_edges(idx: cr.RepoIndex, tree_files: Sequence[Path]) -> dict[str, set[str]]:
    """importer_module -> {launched_module} from real subprocess path sites."""
    edges: dict[str, set[str]] = {}
    for path in tree_files:
        rel_p = cr.rel(path)
        launched = cr.module_name_of(path)
        sites = cr._subprocess_path_sites(rel_p, idx.files, exclude_files=(path,))
        for s in sites:
            if not is_production_path(Path(s.file)):
                continue
            importer = cr.module_name_of(REPO / s.file)
            edges.setdefault(importer, set()).add(launched)
    return edges


def assemble_inventory() -> dict[str, Any]:
    extension_notes = install_engine_extensions()
    idx = cr.build_repo_index()
    engine_doc = cr.assemble()

    tree_files: list[Path] = []
    for root in TREE_ROOTS:
        tree_files.extend(discover_tree_modules(root))

    bindings = recover_orchestration_bindings(idx)
    dynamic_imports = build_dynamic_import_index(idx)
    graph = build_import_graph(idx)
    sub_edges = _subprocess_edges(idx, tree_files)
    reachable, parent = hcli_reachable_set(idx, graph, sub_edges)

    modules: dict[str, dict[str, Any]] = {}
    for path in tree_files:
        rel_p = cr.rel(path)
        dotted = cr.module_name_of(path)
        text = cr.read_text(path)
        shape = module_shape(text, path.name)
        sites = list(cr.find_module_import_sites(idx, dotted, exclude_files=(path,)))
        sites += cr._subprocess_path_sites(rel_p, idx.files, exclude_files=(path,))
        sites += list(dynamic_imports.get(dotted, ()))
        cap = cr.build_capability(
            dotted,
            "module",
            defined=True,
            registered=(path.name in bindings),
            resident_visible=None,
            sites=sites,
            definition={"file": rel_p, "line": None},
        )
        test_sites = [
            cr.Site(s["file"], s["line"], s["kind"]) for s in cap["test_only_sites"]
        ]
        test_info = analyze_test(dotted, path, idx, test_sites)

        hcli = dotted in reachable or any(
            str(s["file"]).startswith("hcli/") for s in cap["call_sites"]
        )
        row: dict[str, Any] = {
            "module": rel_p,
            "dotted": dotted,
            "tree": next((t for t in TREE_ROOTS if rel_p.startswith(t + "/")), TREE_ROOTS[0]),
            "summary": module_summary(text),
            "callable_outside_tests": bool(cap["callable"]),
            "call_sites": cap["call_sites"],
            "test_only_sites": cap["test_only_sites"],
            "hcli_reachable": bool(hcli),
            "hcli_path": cite_hcli_path(dotted, cap["call_sites"], parent),
            "orchestration_bound": path.name in bindings,
            "has_main": shape["has_main"],
            "produces_receipt": shape["produces_receipt"],
            "receipt_names": shape["receipt_names"],
            "receipt_producer_called": bool(shape["produces_receipt"] and cap["callable"]),
            "is_stub": shape["is_stub"],
            "is_package_marker": shape["is_package_marker"],
            "retired": shape["retired"],
            "n_functions": shape["n_functions"],
            "n_classes": shape["n_classes"],
            "evidence_tier": "STATIC",
            **test_info,
        }
        decide(row)
        modules[rel_p] = row

    undisposed = [
        m for m, r in modules.items() if r.get("disposition") not in DISPOSITIONS
    ]
    by_class: dict[str, int] = {c: 0 for c in CLASSIFICATIONS}
    by_disp: dict[str, int] = {d: 0 for d in DISPOSITIONS}
    by_tree: dict[str, int] = {t: 0 for t in TREE_ROOTS}
    for r in modules.values():
        by_class[r["classification"]] = by_class.get(r["classification"], 0) + 1
        by_disp[r["disposition"]] = by_disp.get(r["disposition"], 0) + 1
        by_tree[r["tree"]] = by_tree.get(r["tree"], 0) + 1

    transitions = state_transition_sweep(idx, tree_files)

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Authoritative reachability inventory for every non-test Python "
            "module under tools/future, tools/accelerator, tools/odyssey, "
            "tools/headless. A definition is not a capability."
        ),
        "law": (
            "A capability nothing calls does not exist. Grep for call sites, "
            "not definitions. Own-test-only is not wired. receipts/ are data."
        ),
        "evidence_tier": "STATIC",
        "engine": {
            "analyzer": "tools/future/capability_reachability.py",
            "assemble_used": True,
            "extensions": extension_notes,
            "sidecar_counts": engine_doc.get("counts"),
        },
        "counts": {
            "modules": len(modules),
            "undispositioned": len(undisposed),
            "by_classification": by_class,
            "by_disposition": by_disp,
            "by_tree": by_tree,
            "hcli_reachable": sum(1 for r in modules.values() if r["hcli_reachable"]),
            "callable_outside_tests": sum(
                1 for r in modules.values() if r["callable_outside_tests"]
            ),
            "orchestration_bound": sum(
                1 for r in modules.values() if r["orchestration_bound"]
            ),
            "produces_receipt": sum(1 for r in modules.values() if r["produces_receipt"]),
            "state_transition_findings": len(transitions),
            "state_transition_exits_missing": sum(
                1 for t in transitions if t.get("status") == "EXIT_MISSING"
            ),
        },
        "undispositioned": undisposed,
        "modules": modules,
        "state_transitions": transitions,
        "method": (
            "STATIC source analysis. AST import/call graph + subprocess path "
            "sites + importlib.import_module literals, over every git-tracked "
            "*.py file (sparse-missing files loaded from HEAD). Cannot see a "
            "model-proposed WorkUnit.tool string; callable=false means no "
            "deterministic path, not 'never invoked'."
        ),
    }
    return doc


def build() -> Path:
    return write_receipt(RECEIPT, assemble_inventory(), RECORDED_BY)


def _git_grep(pattern: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "grep",
            "-nE",
            pattern,
            "HEAD",
            "--",
            "*.py",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
    )
    return proc.stdout


_IMPORT_OR_LAUNCH = re.compile(
    r"""(?:^|[\s\(\[])(?:from\s+\S+\s+import|import\s+\S+|subprocess\.(?:run|Popen|check_call|check_output|call)\()"""
)


def _spot_production_hits(rg_out: str, self_path: str, *, import_shaped: bool) -> list[str]:
    hits: list[str] = []
    for line in rg_out.splitlines():
        if not line.startswith("HEAD:"):
            continue
        rest = line[5:]
        path, sep, tail = rest.partition(":")
        if not sep:
            continue
        _, _, text = tail.partition(":")
        if path == self_path:
            continue
        if cr.is_test_path(Path(path)) or is_data_path(Path(path)):
            continue
        if import_shaped and not _IMPORT_OR_LAUNCH.search(text):
            continue
        hits.append(line)
    return hits


def _pick_rows(
    modules: Mapping[str, Mapping[str, Any]],
    classification: str,
    disposition: str,
    n: int,
) -> list[str]:
    rows = [
        (name, row)
        for name, row in modules.items()
        if row.get("classification") == classification and row.get("disposition") == disposition
    ]
    rows.sort(key=lambda kv: kv[0])
    picked: list[str] = []
    seen_trees: set[str] = set()
    for name, row in rows:
        tree = str(row.get("tree") or "")
        if tree not in seen_trees:
            picked.append(name)
            seen_trees.add(tree)
        if len(picked) >= n:
            return picked
    for name, _row in rows:
        if name not in picked:
            picked.append(name)
        if len(picked) >= n:
            break
    return picked


def run_spotchecks(doc: Mapping[str, Any]) -> dict[str, Any]:
    """5 UNREACHABLE + 5 CONNECTED, each verified with git grep -nE against HEAD.

    Evidence tier STATIC. A production import-shaped hit on an UNREACHABLE
    row is a tool bug. A CONNECTED row with no production reference is too.
    """
    modules = doc.get("modules") or {}
    unreachable = _pick_rows(modules, "UNREACHABLE", "PARKED", 5)
    connected = _pick_rows(modules, "BUILT", "CONNECTED", 5)
    if len(unreachable) < 5:
        raise AssertionError(f"not enough UNREACHABLE rows to spot-check: {unreachable}")
    if len(connected) < 5:
        raise AssertionError(f"not enough CONNECTED rows to spot-check: {connected}")
    checks: list[dict[str, Any]] = []
    for name in unreachable:
        dotted = str(modules[name]["dotted"])
        stem = Path(name).stem
        pkg = dotted.rsplit(".", 1)[0]
        pattern = (
            rf"{re.escape(dotted)}"
            rf"|from {re.escape(pkg)} import {re.escape(stem)}"
            rf"|{re.escape(name)}"
        )
        cmd = f"git --no-optional-locks grep -nE {pattern!r} HEAD -- '*.py'"
        out = _git_grep(pattern)
        hits = _spot_production_hits(out, name, import_shaped=True)
        if hits:
            raise AssertionError(
                f"{name} classified UNREACHABLE but git grep -nE found production "
                f"import/launch sites:\n" + "\n".join(hits[:12])
            )
        if modules[name].get("call_sites"):
            raise AssertionError(f"{name} UNREACHABLE but inventory lists call_sites")
        checks.append(
            {
                "verdict": "UNREACHABLE",
                "module": name,
                "command": cmd,
                "n_matches": len(out.splitlines()) if out else 0,
                "production_import_hits": hits,
                "output_head": "\n".join(out.splitlines()[:40]),
            }
        )
    for name in connected:
        dotted = str(modules[name]["dotted"])
        pattern = re.escape(dotted)
        cmd = f"git --no-optional-locks grep -nE {pattern!r} HEAD -- '*.py'"
        out = _git_grep(pattern)
        hits = _spot_production_hits(out, name, import_shaped=False)
        hcli_hits = [h for h in (out.splitlines() if out else []) if h.startswith("HEAD:hcli/")]
        if not modules[name].get("hcli_reachable"):
            raise AssertionError(f"{name} CONNECTED but hcli_reachable is false")
        if not modules[name].get("call_sites"):
            raise AssertionError(f"{name} CONNECTED with no call_sites")
        if not hits and not modules[name].get("hcli_path"):
            raise AssertionError(
                f"{name} CONNECTED but git grep -nE found no production reference to {dotted}"
            )
        checks.append(
            {
                "verdict": "CONNECTED",
                "module": name,
                "command": cmd,
                "n_matches": len(out.splitlines()) if out else 0,
                "hcli_path": modules[name].get("hcli_path"),
                "hcli_hits": hcli_hits[:20],
                "output_head": "\n".join((out.splitlines() if out else [])[:40]),
            }
        )
    return {
        "schema": "hawking.audit.reachability_triage_spotchecks.v1",
        "version": 1,
        "evidence_tier": "STATIC",
        "purpose": "Mandatory 5 UNREACHABLE + 5 CONNECTED hand verification via git grep -nE",
        "unreachable": unreachable,
        "connected": connected,
        "checks": checks,
    }


def selftest() -> Path:
    out = build()
    doc = load_json(out)
    if doc.get("schema") != SCHEMA:
        raise AssertionError(f"schema drifted: {doc.get('schema')!r}")
    if not doc.get("seal_sha256"):
        raise AssertionError("receipt is unsealed")
    modules = doc.get("modules") or {}
    if not modules:
        raise AssertionError("inventory is empty")
    undisposed = [
        name
        for name, row in modules.items()
        if row.get("disposition") not in DISPOSITIONS
    ]
    if undisposed:
        raise AssertionError(
            f"count(modules with no disposition) == {len(undisposed)} (must be 0): "
            + ", ".join(undisposed[:20])
        )
    if int((doc.get("counts") or {}).get("undispositioned") or 0) != 0:
        raise AssertionError("counts.undispositioned is not 0")
    for name, row in modules.items():
        if row.get("classification") not in CLASSIFICATIONS:
            raise AssertionError(f"{name} classification {row.get('classification')!r}")
        if row.get("evidence_tier") != "STATIC":
            raise AssertionError(f"{name} evidence_tier is not STATIC")
        if row["disposition"] == "CONNECTED" and not row.get("hcli_reachable"):
            raise AssertionError(f"{name} CONNECTED but not hcli_reachable")
        if row["disposition"] == "PARKED" and not row.get("wake_condition"):
            raise AssertionError(f"{name} PARKED with no wake_condition")
        if row["disposition"] == "ARCHIVE_CANDIDATE" and not row.get("archive_reason"):
            raise AssertionError(f"{name} ARCHIVE_CANDIDATE with no reason")
        if row.get("callable_outside_tests") and not row.get("call_sites"):
            raise AssertionError(f"{name} callable with no call_sites")
        if (not row.get("callable_outside_tests")) and row.get("call_sites"):
            raise AssertionError(f"{name} has call_sites but callable_outside_tests=false")
    transitions = doc.get("state_transitions") or []
    if not transitions:
        raise AssertionError("state_transitions is empty")
    for t in transitions:
        if not t.get("file") or not t.get("enter"):
            raise AssertionError(f"state-transition missing file/enter: {t!r}")
        if t.get("status") not in {"EXIT_EXISTS", "EXIT_MISSING"}:
            raise AssertionError(f"state-transition status {t.get('status')!r}")
    spot = run_spotchecks(doc)
    write_receipt("REACHABILITY_TRIAGE_SPOTCHECKS.json", spot, RECORDED_BY)
    if len(spot["checks"]) != 10:
        raise AssertionError(f"expected 10 spot-checks, got {len(spot['checks'])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-module reachability inventory with a disposition on every row"
    )
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    require_known_flags({"--build", "--selftest"})
    args = ap.parse_args()
    out = selftest() if args.selftest else build()
    doc = load_json(out)
    counts = doc.get("counts") or {}
    print(out)
    print(
        "modules={n} undisposed={u} CONNECTED={c} PARKED={p} ARCHIVE={a} "
        "BUILT={b} DORMANT={d} UNREACHABLE={un} SCAFFOLDED={s} "
        "hcli_reachable={h} missing_exits={me}".format(
            n=counts.get("modules"),
            u=counts.get("undispositioned"),
            c=(counts.get("by_disposition") or {}).get("CONNECTED"),
            p=(counts.get("by_disposition") or {}).get("PARKED"),
            a=(counts.get("by_disposition") or {}).get("ARCHIVE_CANDIDATE"),
            b=(counts.get("by_classification") or {}).get("BUILT"),
            d=(counts.get("by_classification") or {}).get("DORMANT"),
            un=(counts.get("by_classification") or {}).get("UNREACHABLE"),
            s=(counts.get("by_classification") or {}).get("SCAFFOLDED"),
            h=counts.get("hcli_reachable"),
            me=counts.get("state_transition_exits_missing"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
