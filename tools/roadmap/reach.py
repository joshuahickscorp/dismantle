"""Call-site evidence, reused from tools.future.capability_reachability.

This is not a second analyzer. It patches that module's reader so sparse-checkout
holes and mutation overlays are visible, then calls its import/call/subprocess
helpers (and assemble() when requested).

Import sites are collected, but they are not invocations. A subprocess hit
counts only when the string constant is exactly the gate's own path -- a
suffix match against a different tree (`tools/haider/hcli/scheduler.py` for
`hcli/scheduler.py`) is not a launch of this capability.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools.roadmap.gitfs import REPO, SourceView, classify_symbol

# Imported lazily after the reader is patched so assemble() sees git-backed text.

BUILT_KINDS = frozenset({"call", "subprocess"})
_WEAK_KIND = "weak_signal"


def _load_cr():
    from tools.future import capability_reachability as cr

    return cr


def install_view(view: SourceView):
    """Point capability_reachability.read_text at this SourceView and drop its cache."""
    cr = _load_cr()
    cr._TEXT_CACHE.clear()

    def read_text(path: Path) -> str:
        try:
            rel = path.resolve().relative_to(REPO).as_posix()
        except ValueError:
            rel = str(path)
        text = view.read(rel)
        cr._TEXT_CACHE[path] = text
        return text

    cr.read_text = read_text  # type: ignore[assignment]
    return cr


def _is_test(rel: str) -> bool:
    cr = _load_cr()
    return cr.is_test_path(Path(rel))


def _candidate_files(view: SourceView, needles: Iterable[str]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for needle in needles:
        if not needle:
            continue
        for rel in view.grep_files(needle):
            if rel in seen:
                continue
            seen.add(rel)
            out.append(REPO / rel)
        # Overlay-only files may not be in git grep.
        for rel, text in view.overlay.items():
            if rel in seen:
                continue
            if needle in text:
                seen.add(rel)
                out.append(REPO / rel)
    return out


def module_sites(
    view: SourceView,
    module_dotted: str,
    def_rel: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Production and test sites for a module, using CR's AST import + subprocess helpers."""
    cr = install_view(view)
    needles = [module_dotted]
    stem = module_dotted.rsplit(".", 1)[-1]
    parent = module_dotted.rsplit(".", 1)[0] if "." in module_dotted else ""
    # Relative imports (`from .scheduler import`) never mention the dotted
    # name. Include the precise forms CR's AST resolver will accept.
    needles.extend(
        [
            f"from .{stem} import",
            f"from {module_dotted} import",
            f"import {module_dotted}",
        ]
    )
    if parent:
        needles.append(f"from {parent} import {stem}")
    if def_rel:
        needles.append(def_rel)
        needles.append(f"import {stem}")
    files = _candidate_files(view, needles)
    if def_rel:
        def_path = REPO / def_rel
        if def_path not in files:
            files.append(def_path)
    if not files:
        return [], []
    idx = cr.build_repo_index(files=files)
    exclude = (REPO / def_rel,) if def_rel else ()
    sites = list(cr.find_module_import_sites(idx, module_dotted, exclude_files=exclude))
    if def_rel:
        sites.extend(cr._subprocess_path_sites(def_rel, idx.files, exclude_files=exclude))
    prod, test = cr._partition(sites)
    return [s.to_dict() for s in prod], [s.to_dict() for s in test]


def symbol_sites(
    view: SourceView,
    module_dotted: str,
    symbol: str,
    def_rel: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cr = install_view(view)
    needles = [module_dotted, symbol]
    if def_rel:
        needles.append(def_rel)
    files = _candidate_files(view, needles)
    if def_rel:
        def_path = REPO / def_rel
        if def_path not in files:
            files.append(def_path)
    if not files:
        return [], []
    idx = cr.build_repo_index(files=files)
    exclude = (REPO / def_rel,) if def_rel else ()
    sites = cr.find_symbol_call_sites(idx, module_dotted, symbol, exclude_files=exclude)
    prod, test = cr._partition(sites)
    return [s.to_dict() for s in prod], [s.to_dict() for s in test]


def assemble_snapshot(view: SourceView) -> dict[str, Any]:
    """Reuse capability_reachability.assemble() against the current view."""
    cr = install_view(view)

    orig_repo_py = cr.repo_py_files

    def repo_py_files() -> list[Path]:
        return [REPO / rel for rel in view.tracked_py()]

    cr.repo_py_files = repo_py_files  # type: ignore[assignment]
    try:
        doc = cr.assemble()
    finally:
        cr.repo_py_files = orig_repo_py  # type: ignore[assignment]
    return {
        "schema": doc.get("schema"),
        "counts": doc.get("counts"),
        "dead_surface_count": len(doc.get("DEAD_SURFACE") or []),
        "method": doc.get("method"),
        "law": doc.get("law"),
    }


def is_test_path(rel: str) -> bool:
    return _is_test(rel)


def is_exact_cli_path(rel_path: str, value: str) -> bool:
    """True iff `value` is a launch of this repo-relative path, not a suffix of another.

    `tools/haider/hcli/scheduler.py` is not a launch of `hcli/scheduler.py`.
    Substring/suffix matching is how the previous auditor credited five
    AgentOS gates with one grep of a different tree.
    """
    if not rel_path or not value:
        return False
    v = value.replace("\\", "/").strip()
    if v == rel_path or v == "./" + rel_path:
        return True
    repo_posix = REPO.as_posix().rstrip("/")
    if v == repo_posix + "/" + rel_path:
        return True
    try:
        resolved = REPO.resolve().as_posix().rstrip("/")
    except OSError:
        resolved = repo_posix
    return v == resolved + "/" + rel_path


def _pairs(probe: dict[str, Any]) -> list[tuple[str | None, str | None]]:
    paths: list[str] = list(probe.get("code_paths") or [])
    modules: list[str] = list(probe.get("modules") or [])
    if paths and modules and len(paths) == len(modules):
        return list(zip(modules, paths))
    if not modules and paths:
        return [
            (Path(p).with_suffix("").as_posix().replace("/", "."), p) for p in paths
        ]
    if modules and not paths:
        return [(m, m.replace(".", "/") + ".py") for m in modules]
    n = max(len(paths), len(modules))
    out: list[tuple[str | None, str | None]] = []
    for i in range(n):
        out.append(
            (
                modules[i] if i < len(modules) else None,
                paths[i] if i < len(paths) else None,
            )
        )
    return out


def unique_code_paths(table: dict[str, dict[str, Any]]) -> set[str]:
    """Paths listed by exactly one probe in `table`. Shared modules cannot
    prove two different gates by the same CLI launch or auto-discovered API."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for probe in table.values():
        for path in probe.get("code_paths") or []:
            counts[path] += 1
    return {path for path, n in counts.items() if n == 1}


def _import_needles(module_dotted: str, def_rel: str | None) -> list[str]:
    needles = [module_dotted]
    stem = module_dotted.rsplit(".", 1)[-1]
    parent = module_dotted.rsplit(".", 1)[0] if "." in module_dotted else ""
    needles.extend(
        [
            f"from .{stem} import",
            f"from {module_dotted} import",
            f"import {module_dotted}",
        ]
    )
    if parent:
        needles.append(f"from {parent} import {stem}")
    if def_rel:
        needles.append(def_rel)
    return needles


def _strict_subprocess_sites(
    cr: Any,
    rel_path: str,
    files: Sequence[Path],
    *,
    exclude_files: Iterable[Path] = (),
) -> list[Any]:
    """Like CR._subprocess_path_sites but the path string must be this file."""
    excluded = {cr.rel(p) for p in exclude_files}
    sites: list[Any] = []
    seen: set[tuple[str, int]] = set()
    for path in files:
        rp = cr.rel(path)
        if rp in excluded or rp == rel_path:
            continue
        text = cr.read_text(path)
        if rel_path not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and cr._is_subprocess_call(node)):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Constant)
                    and isinstance(inner.value, str)
                    and is_exact_cli_path(rel_path, inner.value)
                ):
                    key = (rp, int(node.lineno))
                    if key in seen:
                        break
                    seen.add(key)
                    sites.append(cr.Site(rp, int(node.lineno), "subprocess"))
                    break
    return sites


def _weak_name_sites(
    cr: Any,
    idx: Any,
    module_dotted: str,
    symbol: str,
    *,
    exclude_files: Iterable[Path] = (),
) -> list[Any]:
    """Bound Name/Attribute mentions of `symbol` that are not Call-of-that-name.

    `except NO_PROGRESS`, `x = NO_PROGRESS`, a string, a comment -- none of
    these invoke the capability. Recorded as weak_signal; they never move status.
    """
    excluded = {cr.rel(p) for p in exclude_files}
    sites: list[Any] = []
    target_full = f"{module_dotted}.{symbol}"
    for path in idx.files:
        rp = cr.rel(path)
        if rp in excluded:
            continue
        binds = idx.bound_names.get(rp, [])
        local_direct = {local for local, full in binds if full == target_full}
        local_module = {local for local, full in binds if full == module_dotted}
        if not local_direct and not local_module:
            continue
        text = cr.read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        call_func_ids: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in local_direct:
                call_func_ids.add(id(func))
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == symbol
                and isinstance(func.value, ast.Name)
                and func.value.id in local_module
            ):
                call_func_ids.add(id(func))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in local_direct:
                if id(node) in call_func_ids:
                    continue
                if isinstance(getattr(node, "ctx", None), ast.Store):
                    continue
                sites.append(cr.Site(rp, int(node.lineno), _WEAK_KIND))
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == symbol
                and isinstance(node.value, ast.Name)
                and node.value.id in local_module
            ):
                if id(node) in call_func_ids:
                    continue
                sites.append(cr.Site(rp, int(node.lineno), _WEAK_KIND))
    return sites


def _attach(site: Any, *, symbol: str | None = None) -> dict[str, Any]:
    row = site.to_dict() if hasattr(site, "to_dict") else dict(site)
    if symbol and "symbol" not in row:
        row["symbol"] = symbol
    return row


def scan_probe(
    view: SourceView,
    probe: dict[str, Any],
    *,
    unique_paths: set[str],
) -> dict[str, Any]:
    """Collect evidence for one catalog probe.

    runtime_caller holds only kind=call and kind=subprocess of THIS probe's
    implementing symbol(s) (or a unique-path CLI launch). Imports are
    import_sites. Name-only matches are weak_signals.
    """
    cr = install_view(view)
    pairs = _pairs(probe)
    needles: list[str] = []
    for module, path in pairs:
        if module:
            needles.extend(_import_needles(module, path))
        elif path:
            needles.append(path)

    catalog_symbols: list[dict[str, str]] = [dict(s) for s in (probe.get("symbols") or [])]
    # Do not auto-discover every public name in a uniquely-owned module.
    # BUILT binds to the catalogued implementing symbol (or an exact CLI
    # launch of a uniquely-owned path). Guessing `main` is not a gate.

    files = _candidate_files(view, needles)
    for _module, path in pairs:
        if path:
            def_path = REPO / path
            if def_path not in files:
                files.append(def_path)

    defined_refs: list[dict[str, Any]] = []
    missing: list[str] = []
    for _module, path in pairs:
        if not path:
            continue
        if view.exists(path):
            text = view.read(path)
            defined_refs.append(
                {"file": path, "line": 1 if text else None, "kind": "definition"}
            )
        else:
            missing.append(path)

    if not files:
        return {
            "defined": bool(defined_refs),
            "defined_refs": defined_refs,
            "missing_paths": missing,
            "runtime_caller": [],
            "import_sites": [],
            "weak_signals": [],
            "tests": [],
            "symbols_scanned": [s.get("symbol") for s in catalog_symbols],
        }

    idx = cr.build_repo_index(files=files)

    import_prod: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    seen_imp: set[tuple[str, int, str]] = set()
    seen_test: set[tuple[str, int, str]] = set()
    for module, path in pairs:
        if not module:
            continue
        exclude = (REPO / path,) if path else ()
        sites = list(cr.find_module_import_sites(idx, module, exclude_files=exclude))
        prod, test = cr._partition(sites)
        for s in prod:
            key = (s.file, s.line, s.kind)
            if key in seen_imp:
                continue
            seen_imp.add(key)
            import_prod.append(_attach(s))
        for s in test:
            key = (s.file, s.line, s.kind)
            if key in seen_test:
                continue
            seen_test.add(key)
            tests.append(_attach(s))

    specs = catalog_symbols
    runtime: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    seen_run: set[tuple[str, int, str, str]] = set()
    seen_weak: set[tuple[str, int, str]] = set()

    for spec in specs:
        module = spec["module"]
        symbol = spec["symbol"]
        def_rel = None
        for m, p in pairs:
            if m == module:
                def_rel = p
                break
        kind = spec.get("kind")
        line: int | None = None
        if def_rel and view.exists(def_rel):
            classified, line = classify_symbol(view.read(def_rel), symbol)
            kind = classified or kind
            if kind in ("function", "class") and line:
                defined_refs.append(
                    {"file": def_rel, "line": line, "kind": "symbol", "note": symbol}
                )
            elif kind == "assignment":
                key = (def_rel, int(line or 0), _WEAK_KIND)
                if key not in seen_weak:
                    seen_weak.add(key)
                    weak.append(
                        {
                            "file": def_rel,
                            "line": line,
                            "kind": _WEAK_KIND,
                            "symbol": symbol,
                            "note": "name-only assignment; not an invocable implementing symbol",
                        }
                    )
                continue

        exclude = (REPO / def_rel,) if def_rel else ()
        call_sites = cr.find_symbol_call_sites(
            idx, module, symbol, exclude_files=exclude
        )
        prod, test = cr._partition(call_sites)
        for s in prod:
            key = (s.file, s.line, s.kind, symbol)
            if key in seen_run:
                continue
            seen_run.add(key)
            runtime.append(_attach(s, symbol=symbol))
        for s in test:
            key = (s.file, s.line, s.kind)
            if key in seen_test:
                continue
            seen_test.add(key)
            tests.append(_attach(s, symbol=symbol))

        weak_sites = _weak_name_sites(cr, idx, module, symbol, exclude_files=exclude)
        weak_prod, _weak_test = cr._partition(weak_sites)
        for s in weak_prod:
            key = (s.file, s.line, s.kind)
            if key in seen_weak:
                continue
            seen_weak.add(key)
            weak.append(_attach(s, symbol=symbol))

    for module, path in pairs:
        if not path or path not in unique_paths:
            continue
        exclude = (REPO / path,)
        sub_sites = _strict_subprocess_sites(cr, path, idx.files, exclude_files=exclude)
        prod, test = cr._partition(sub_sites)
        for s in prod:
            key = (s.file, s.line, s.kind, path)
            if key in seen_run:
                continue
            seen_run.add(key)
            runtime.append(_attach(s, symbol=path))
        for s in test:
            key = (s.file, s.line, s.kind)
            if key in seen_test:
                continue
            seen_test.add(key)
            tests.append(_attach(s, symbol=path))

    runtime.sort(key=lambda s: (s.get("file") or "", s.get("line") or 0, s.get("kind") or ""))
    import_prod.sort(key=lambda s: (s.get("file") or "", s.get("line") or 0))
    weak.sort(key=lambda s: (s.get("file") or "", s.get("line") or 0))
    tests.sort(key=lambda s: (s.get("file") or "", s.get("line") or 0))

    return {
        "defined": bool(defined_refs),
        "defined_refs": defined_refs,
        "missing_paths": missing,
        "runtime_caller": runtime,
        "import_sites": import_prod,
        "weak_signals": weak[:24],
        "tests": tests,
        "symbols_scanned": [s.get("symbol") for s in specs],
    }
