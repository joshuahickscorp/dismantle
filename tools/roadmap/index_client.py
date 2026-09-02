"""Client for the hawking-index python-facts JSON surface.

Lane r1 may later expose the same command on the `hawking-index` binary.
This client accepts either:

    hawking-index-query python-facts --git-head --commit <sha> --repo <repo>
    hawking-index python-facts --git-head --commit <sha> --repo <repo>

Schema: hawking.index.python_facts.v1

The dump is built once per SourceView (overlay included) from the named git
commit's blobs (default HEAD), never the working tree. A sparse worktree
where hcli/ is absent from disk still indexes those files. Untracked and
uncommitted files are invisible. Every file fact carries the commit it was
parsed from.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.roadmap.gitfs import REPO, SourceView, head_commit

SCHEMA = "hawking.index.python_facts.v1"

# Set ROADMAP_REACH_BACKEND=ast to force the old capability_reachability path.
# index = require the rust dump; auto = index if the binary exists else ast.
_BACKEND_ENV = "ROADMAP_REACH_BACKEND"


def backend() -> str:
    raw = (os.environ.get(_BACKEND_ENV) or "auto").strip().lower()
    if raw in {"ast", "cr", "python"}:
        return "ast"
    if raw in {"index", "rust"}:
        return "index"
    return "auto"


def _bin_candidates() -> list[Path]:
    out: list[Path] = []
    for key in ("HAWKING_INDEX_QUERY_BIN", "HAWKING_INDEX_BIN"):
        env = os.environ.get(key)
        if env:
            out.append(Path(env))
    roots = [
        REPO / "workspace" / "ops" / "build" / "rust",
        REPO / "target",
    ]
    names = ("hawking-index-query", "hawking-index")
    for root in roots:
        for kind in ("release", "debug"):
            for name in names:
                out.append(root / kind / name)
    which = shutil.which("hawking-index-query") or shutil.which("hawking-index")
    if which:
        out.append(Path(which))
    return out


def find_index_bin() -> Path | None:
    for cand in _bin_candidates():
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def resolve_backend() -> str:
    choice = backend()
    if choice == "ast":
        return "ast"
    if choice == "index":
        if find_index_bin() is None:
            raise FileNotFoundError(
                "ROADMAP_REACH_BACKEND=index but hawking-index-query (or "
                "hawking-index) is not built. cargo build -p hawking-index-query "
                "--release (CARGO_TARGET_DIR=workspace/ops/build/rust)"
            )
        return "index"
    return "index" if find_index_bin() is not None else "ast"


def _overlay_ndjson(view: SourceView) -> str:
    lines = []
    for rel, text in view.overlay.items():
        if not rel.endswith(".py"):
            continue
        lines.append(json.dumps({"path": rel, "content": text}, ensure_ascii=False))
    return "\n".join(lines)


def catalog_watch_names() -> list[str]:
    """Symbol / module-stem names the auditor will query. Shrinks the dump."""
    from tools.roadmap import catalog

    names: set[str] = set()
    for table in (catalog.GATES, catalog.GENES):
        for probe in table.values():
            for spec in probe.get("symbols") or []:
                if spec.get("symbol"):
                    names.add(spec["symbol"])
            for mod in probe.get("modules") or []:
                names.add(mod.rsplit(".", 1)[-1])
                names.add(mod)
            for path in probe.get("code_paths") or []:
                names.add(Path(path).stem)
    return sorted(n for n in names if n)


def load_python_facts(view: SourceView) -> dict[str, Any]:
    """Run the rust dump against this view. Result is cached on the view."""
    cached = getattr(view, "_python_facts", None)
    if cached is not None:
        return cached
    bin_path = find_index_bin()
    if bin_path is None:
        raise FileNotFoundError("hawking-index-query binary not found")
    commit = head_commit()
    cmd = [
        str(bin_path),
        "python-facts",
        "--git-head",
        "--commit",
        commit,
        "--repo",
        str(REPO),
    ]
    for name in catalog_watch_names():
        cmd.extend(["--watch", name])
    overlay = _overlay_ndjson(view)
    cp = subprocess.run(
        cmd,
        input=overlay,
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} exited {cp.returncode}: {cp.stderr[-4000:]}"
        )
    dump = json.loads(cp.stdout)
    if dump.get("schema") != SCHEMA:
        raise RuntimeError(
            f"python-facts schema {dump.get('schema')!r} != {SCHEMA}; "
            "r1/r2 JSON surfaces drifted"
        )
    dump_commit = dump.get("commit") or commit
    by_path = {f["path"]: f for f in dump.get("files") or [] if f.get("path")}
    wrapped = {
        "schema": dump["schema"],
        "commit": dump_commit,
        "files": by_path,
        "file_count": len(by_path),
        "bin": str(bin_path),
    }
    view._python_facts = wrapped  # type: ignore[attr-defined]
    return wrapped


def facts_for(view: SourceView) -> dict[str, Any] | None:
    if resolve_backend() != "index":
        return None
    return load_python_facts(view)


def warmup(view: SourceView) -> dict[str, Any] | None:
    return facts_for(view)


def file_facts(dump: dict[str, Any], rel: str) -> dict[str, Any] | None:
    files = dump.get("files") or {}
    return files.get(rel)


def module_name_of_rel(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolved_from_modules(importer_rel: str, imp: dict[str, Any]) -> list[str]:
    """Port of tools.future.capability_reachability._resolved_from_modules."""
    importer = REPO / importer_rel
    bases: list[str] = []
    level = int(imp.get("level") or 0)
    if imp.get("form") == "from" and level > 0:
        importer_mod = module_name_of_rel(importer_rel)
        parts = importer_mod.split(".") if importer_mod else []
        is_init = Path(importer_rel).name == "__init__.py"
        base_parts = parts if is_init else parts[:-1]
        if level > 1:
            cut = level - 1
            base_parts = base_parts[: max(0, len(base_parts) - cut)]
        base = ".".join(base_parts)
        mod = f"{base}.{imp['module']}" if imp.get("module") else base
        if mod:
            bases.append(mod)
    else:
        mod = imp.get("module") or ""
        if mod:
            bases.append(mod)
            if "." not in mod:
                sib = importer.parent / f"{mod}.py"
                if sib.is_file() and sib != importer:
                    try:
                        rel = sib.resolve().relative_to(REPO).as_posix()
                    except ValueError:
                        rel = str(sib)
                    bases.append(module_name_of_rel(rel))
    return bases


def import_targets_and_binds(
    importer_rel: str, imp: dict[str, Any]
) -> tuple[list[str], list[tuple[str, str]]]:
    """Port of CR import-target + bound_names construction for one statement."""
    targets: list[str] = []
    binds: list[tuple[str, str]] = []
    importer = REPO / importer_rel
    if imp.get("form") == "import":
        for alias in imp.get("names") or []:
            name = alias.get("name") or ""
            if not name:
                continue
            asname = alias.get("asname")
            targets.append(name)
            sib = importer.parent / (name.split(".")[0] + ".py")
            if "." not in name and sib.is_file() and sib != importer:
                try:
                    rel = sib.resolve().relative_to(REPO).as_posix()
                except ValueError:
                    rel = str(sib)
                targets.append(module_name_of_rel(rel))
            local = asname or name.split(".")[0]
            binds.append((local, name))
    else:
        for mod in _resolved_from_modules(importer_rel, imp):
            targets.append(mod)
            for alias in imp.get("names") or []:
                aname = alias.get("name") or ""
                if not aname:
                    continue
                targets.append(f"{mod}.{aname}")
                local = alias.get("asname") or aname
                binds.append((local, f"{mod}.{aname}"))
    return targets, binds


def classify_from_facts(
    ff: dict[str, Any] | None, symbol: str
) -> tuple[str | None, int | None]:
    """Same order as gitfs.classify_symbol: module-level first, then nested."""
    if not ff or not symbol:
        return None, None
    defs = ff.get("definitions") or []
    for d in defs:
        if d.get("name") == symbol and d.get("scope") == "module":
            kind = d.get("kind")
            line = d.get("line")
            return (str(kind) if kind else None, int(line) if line else None)
    for d in defs:
        if d.get("name") == symbol and d.get("kind") in ("function", "class"):
            kind = d.get("kind")
            line = d.get("line")
            return (str(kind) if kind else None, int(line) if line else None)
    return None, None
