"""Git-backed source view.

This worktree is a sparse checkout. A path missing from disk is not evidence
the file does not exist. Reads go: overlay -> working tree -> `git show HEAD`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        check=check,
    )


class SourceView:
    """Readable snapshot of HEAD plus an optional overlay used by mutation checks."""

    def __init__(self) -> None:
        self.overlay: dict[str, str] = {}
        self._cache: dict[str, str] = {}
        self._exists_cache: dict[str, bool] = {}
        self._grep_cache: dict[str, list[str]] = {}
        self._py_files: list[str] | None = None

    def tracked_py(self) -> list[str]:
        if self._py_files is None:
            out = _git("ls-files", "*.py").stdout.splitlines()
            self._py_files = [line for line in out if line and "__pycache__" not in line]
        files = list(self._py_files)
        for rel in self.overlay:
            if rel.endswith(".py") and rel not in files:
                files.append(rel)
        return files

    def exists(self, rel: str) -> bool:
        if rel in self.overlay:
            return True
        if rel in self._exists_cache:
            return self._exists_cache[rel]
        disk = REPO / rel
        if disk.is_file():
            self._exists_cache[rel] = True
            return True
        cp = _git("cat-file", "-e", f"HEAD:{rel}", check=False)
        present = cp.returncode == 0
        self._exists_cache[rel] = present
        return present

    def read(self, rel: str) -> str:
        if rel in self.overlay:
            return self.overlay[rel]
        if rel in self._cache:
            return self._cache[rel]
        disk = REPO / rel
        if disk.is_file():
            text = disk.read_text(encoding="utf-8", errors="replace")
        else:
            cp = _git("show", f"HEAD:{rel}", check=False)
            if cp.returncode != 0:
                text = ""
            else:
                text = cp.stdout
        self._cache[rel] = text
        return text

    def grep_files(self, needle: str) -> list[str]:
        """Candidate files whose HEAD blob contains `needle`. Overlay files are always included."""
        if not needle:
            return []
        if needle in self._grep_cache:
            hits = list(self._grep_cache[needle])
        else:
            # Search HEAD, not the work tree: this checkout is sparse and a
            # missing-on-disk file is not evidence of a missing caller.
            # Pattern: git grep -F -e <needle> HEAD -- '*.py'
            cp = _git("grep", "-l", "-F", "-e", needle, "HEAD", "--", "*.py", check=False)
            hits = []
            for line in cp.stdout.splitlines():
                if not line:
                    continue
                # `git grep <tree> -l` prefixes each path with `<tree>:`.
                if line.startswith("HEAD:"):
                    line = line[len("HEAD:") :]
                hits.append(line)
            self._grep_cache[needle] = list(hits)
        for rel in self.overlay:
            if rel.endswith(".py") and rel not in hits:
                hits.append(rel)
        return hits

    def path(self, rel: str) -> Path:
        return REPO / rel


def classify_symbol(text: str, symbol: str) -> tuple[str | None, int | None]:
    """Classify a module-level name: function / class / assignment / None.

    A NAME that is only assigned (a constant, a string, a comment) is not an
    invocable implementing symbol. Call-site analysis must not treat it as one.
    """
    import ast

    if not text or not symbol:
        return None, None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _classify_symbol_lines(text, symbol)
    # Module-level first so a constant named like a nested helper cannot
    # masquerade as an invocable, and a method still counts as a function.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            return "function", int(node.lineno)
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            return "class", int(node.lineno)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return "assignment", int(node.lineno)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == symbol
        ):
            return "assignment", int(node.lineno)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            return "function", int(node.lineno)
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            return "class", int(node.lineno)
    return None, None


def _classify_symbol_lines(text: str, symbol: str) -> tuple[str | None, int | None]:
    prefix_def = f"def {symbol}("
    prefix_async = f"async def {symbol}("
    prefix_cls = f"class {symbol}"
    assign = f"{symbol} ="
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(prefix_def) or stripped.startswith(prefix_async):
            return "function", i
        if stripped.startswith(prefix_cls) and (
            len(stripped) == len(prefix_cls)
            or stripped[len(prefix_cls)] in "(: \t"
        ):
            return "class", i
        if stripped.startswith(assign):
            return "assignment", i
    return None, None


def definition_line(text: str, symbol: str) -> int | None:
    """First invocable `def`/`class` line for `symbol`, or None.

    Assignments (constants) are not definitions of a callable capability.
    """
    kind, line = classify_symbol(text, symbol)
    if kind in ("function", "class"):
        return line
    return None


def iter_py_rel(view: SourceView, extra: Iterable[str] = ()) -> list[str]:
    files = list(view.tracked_py())
    for rel in extra:
        if rel not in files:
            files.append(rel)
    return files
