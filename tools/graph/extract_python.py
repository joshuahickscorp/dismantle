"""Python extraction via stdlib ast — full fidelity."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from graph_model import (
    Graph,
    complexity_of,
    detect_security,
    detect_side_effects,
    make_node,
)

# Map top-level import roots that live in-repo
REPO_ROOTS = {
    "tools", "ramanujan", "odyssey", "app", "control", "profiles", "packs",
}


def _module_name_from_path(rel: str) -> str:
    p = Path(rel)
    if p.name == "__init__.py":
        return str(p.parent).replace("/", ".")
    return str(p.with_suffix("")).replace("/", ".")


def _pkg_id_for(rel: str) -> str | None:
    """pkg:<python top-level package path> for tools/foo, ramanujan, etc."""
    parts = Path(rel).parts
    if not parts:
        return None
    if parts[0] in REPO_ROOTS:
        if len(parts) >= 2 and parts[0] == "tools":
            return f"pkg:tools/{parts[1]}"
        return f"pkg:{parts[0]}"
    return None


def _complexity_ast(node: ast.AST) -> int:
    n = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While,
                              ast.ExceptHandler, ast.With, ast.AsyncWith,
                              ast.Assert, ast.comprehension)):
            n += 1
        elif isinstance(child, ast.BoolOp):
            n += max(0, len(child.values) - 1)
        elif isinstance(child, ast.Match):
            n += len(child.cases)
        elif isinstance(child, ast.IfExp):
            n += 1
    return n


def _attr_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attr_chain(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


def _call_name(node: ast.Call) -> str | None:
    return _attr_chain(node.func)


class PyExtractor(ast.NodeVisitor):
    def __init__(
        self,
        rel: str,
        text: str,
        g: Graph,
        indexes: dict[str, Any],
        module_index: dict[str, str],
    ):
        self.rel = rel
        self.text = text
        self.lines = text.splitlines()
        self.g = g
        self.indexes = indexes
        self.module_index = module_index
        self.file_id = f"file:{rel}"
        self.module = _module_name_from_path(rel)
        self.class_stack: list[str] = []
        self.fn_stack: list[str] = []
        self.current_fn_id: str | None = None

    def _span(self, node: ast.AST) -> list[int]:
        start = getattr(node, "lineno", 1) or 1
        end = getattr(node, "end_lineno", start) or start
        return [start, end]

    def _body_text(self, node: ast.AST) -> str:
        span = self._span(node)
        return "\n".join(self.lines[span[0] - 1:span[1]])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = ".".join(self.class_stack + [node.name]) if self.class_stack else node.name
        tid = f"type:{self.rel}#{qname}"
        body = self._body_text(node)
        # Topology-aligned public: module-level class not starting with _
        is_pub = (not self.class_stack) and (not node.name.startswith("_"))
        self.g.add_node(make_node(
            "type", tid, qname, path=self.rel, lang="python",
            span=self._span(node),
            loc=max(1, (node.end_lineno or node.lineno) - node.lineno + 1),
            public=is_pub,
            complexity=_complexity_ast(node),
            side_effects=detect_side_effects(body),
            security_sensitive=detect_security(body),
        ))
        self.g.ensure_contains(self.file_id, tid, evidence="ast")
        self.indexes["types_by_name"][node.name].append(tid)
        self.indexes["types_by_name"][qname].append(tid)
        self.indexes["py_types"][tid] = node.name

        # bases -> implements-like (use implements for class bases)
        for base in node.bases:
            bname = _attr_chain(base)
            if bname:
                cands = self.indexes["types_by_name"].get(bname.split(".")[-1], [])
                if len(cands) == 1:
                    self.g.add_edge(
                        tid, "implements", cands[0],
                        evidence="ast", confidence=0.9,
                    )

        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_fn(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_fn(node)

    def _visit_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.class_stack:
            qname = f"{'.'.join(self.class_stack)}.{node.name}"
        else:
            qname = node.name
        fid = f"fn:{self.rel}#{qname}"
        body = self._body_text(node)
        # Topology-aligned public: only module-level def (col 0), not methods
        is_pub = (not self.class_stack) and (not node.name.startswith("_"))
        is_test = (
            node.name.startswith("test_")
            or any(
                (isinstance(d, ast.Attribute) and d.attr == "test")
                or (isinstance(d, ast.Name) and d.id in ("test", "pytest"))
                for d in node.decorator_list
            )
            or "/tests/" in self.rel
            or Path(self.rel).name.startswith("test_")
        )
        self.g.add_node(make_node(
            "function", fid, qname, path=self.rel, lang="python",
            span=self._span(node),
            loc=max(1, (node.end_lineno or node.lineno) - node.lineno + 1),
            public=is_pub,
            test=is_test,
            complexity=_complexity_ast(node),
            side_effects=detect_side_effects(body),
            security_sensitive=detect_security(body),
        ))
        self.g.ensure_contains(self.file_id, fid, evidence="ast")
        self.indexes["fns_by_name"][node.name].append(fid)
        self.indexes["fns_by_qual"][qname].append(fid)
        self.indexes["fn_meta"][fid] = {
            "crate": None,
            "file": self.rel,
            "name": node.name,
            "qname": qname,
            "body": body,
            "is_test": is_test,
            "lang": "python",
            "calls": [],  # filled below
        }

        if is_test:
            tid = f"test:{self.rel}#{qname}"
            self.g.add_node(make_node(
                "test", tid, qname, path=self.rel, lang="python",
                span=self._span(node),
                loc=max(1, (node.end_lineno or node.lineno) - node.lineno + 1),
                public=False,
                test=True,
                complexity=_complexity_ast(node),
                side_effects=detect_side_effects(body),
                security_sensitive=detect_security(body),
            ))
            self.g.ensure_contains(self.file_id, tid, evidence="ast")
            self.g.add_edge(tid, "tests", fid, evidence="test", confidence=1.0)
            self.indexes["test_nodes"][tid] = fid

        # decorators
        for dec in node.decorator_list:
            dname = _attr_chain(dec) if not isinstance(dec, ast.Call) else _call_name(dec)
            if dname:
                # no dedicated edge; note security/cli later
                pass

        prev = self.current_fn_id
        self.current_fn_id = fid
        self.fn_stack.append(qname)

        # Walk body for calls
        for child in node.body:
            self.visit(child)

        self.fn_stack.pop()
        self.current_fn_id = prev

    def visit_Call(self, node: ast.Call) -> None:
        cname = _call_name(node)
        if cname and self.current_fn_id:
            self.indexes["fn_meta"][self.current_fn_id]["calls"].append(cname)
            short = cname.split(".")[-1]
            # resolve later in batch
            self.indexes["py_calls"].append((self.current_fn_id, cname, short))
            # argparse subparsers
            if short in ("add_parser", "add_subparsers") or "ArgumentParser" in cname:
                self.indexes["argparse_sites"].append((self.rel, self.current_fn_id, node))
            # emit/publish
            low = short.lower()
            if any(x in low for x in ("emit", "publish", "send_event", "subscribe", "on_event")):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        ev = arg.value
                        eid = f"event:{ev}"
                        if eid not in self.g.nodes:
                            self.g.add_node(make_node(
                                "event", eid, ev, path=self.rel, lang="python",
                            ))
                        et = "emits" if any(x in low for x in ("emit", "publish", "send")) else "consumes"
                        self.g.add_edge(
                            self.current_fn_id, et, eid,
                            evidence="ast", confidence=0.7,
                        )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._resolve_import(alias.name, None)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if node.level and node.level > 0:
            # relative
            parts = self.module.split(".")
            base = parts[: max(0, len(parts) - node.level)]
            if mod:
                mod = ".".join(base + mod.split("."))
            else:
                mod = ".".join(base)
        for alias in node.names:
            self._resolve_import(mod, alias.name if alias.name != "*" else None)

    def _resolve_import(self, module: str | None, name: str | None) -> None:
        if not module:
            return
        # map to file/module node
        candidates = [
            module.replace(".", "/") + ".py",
            module.replace(".", "/") + "/__init__.py",
        ]
        # also tools.condense.foo style
        for c in candidates:
            if c in self.module_index:
                dst = self.module_index[c]
                self.g.add_edge(
                    self.file_id, "imports", dst,
                    evidence="ast", confidence=1.0,
                )
                return
        # pkg level
        top = module.split(".")[0]
        for root in REPO_ROOTS:
            if module.startswith(root) or top == root:
                # try file path
                p = module.replace(".", "/") + ".py"
                if p in self.module_index:
                    self.g.add_edge(
                        self.file_id, "imports", self.module_index[p],
                        evidence="ast", confidence=1.0,
                    )
                    return

    def visit_If(self, node: ast.If) -> None:
        # if __name__ == "__main__"
        if self._is_main_guard(node.test):
            cid = f"cli:{self.module} __main__"
            self.g.add_node(make_node(
                "cli_command", cid, f"{self.module} __main__",
                path=self.rel, lang="python",
                span=self._span(node),
                public=True,
            ))
            self.g.ensure_contains(self.file_id, cid, evidence="ast")
            self.indexes["cli_commands"].append(cid)
        self.generic_visit(node)

    def _is_main_guard(self, test: ast.AST) -> bool:
        if isinstance(test, ast.Compare):
            left = test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                for comp in test.comparators:
                    if isinstance(comp, ast.Constant) and comp.value == "__main__":
                        return True
        return False


def extract_python_file(
    repo: Path,
    rel: str,
    g: Graph,
    indexes: dict[str, Any],
    module_index: dict[str, str],
) -> None:
    path = repo / rel
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return

    # ensure pkg node
    pkg = _pkg_id_for(rel)
    if pkg and pkg not in g.nodes:
        g.add_node(make_node(
            "crate", pkg, pkg.removeprefix("pkg:"),
            path=str(Path(rel).parts[0]) if Path(rel).parts else rel,
            lang="python",
            public=True,
            subsystem="laboratory" if rel.startswith(("tools/", "ramanujan/")) else "shared",
        ))
        g.ensure_contains("repo", pkg, evidence="ast")
    if pkg:
        g.ensure_contains(pkg, f"file:{rel}", evidence="ast")

    ex = PyExtractor(rel, text, g, indexes, module_index)
    ex.visit(tree)

    # argparse: scan source text for add_parser("name")
    for m in re.finditer(
        r"""add_parser\(\s*['"]([\w\-]+)['"]""",
        text,
    ):
        sub = m.group(1)
        prog = Path(rel).stem
        cid = f"cli:{prog} {sub}"
        g.add_node(make_node(
            "cli_command", cid, f"{prog} {sub}",
            path=rel, lang="python", public=True,
        ))
        g.ensure_contains(f"file:{rel}", cid, evidence="ast")
        indexes["cli_commands"].append(cid)

    # os.environ / env reads -> feature flags handled in registries


def resolve_python_calls(g: Graph, indexes: dict[str, Any]) -> None:
    for src, cname, short in indexes.get("py_calls", []):
        targets: list[str] = []
        # exact qual
        if cname in indexes["fns_by_qual"]:
            targets = list(dict.fromkeys(indexes["fns_by_qual"][cname]))
        elif short in indexes["fns_by_name"]:
            targets = list(dict.fromkeys(indexes["fns_by_name"][short]))
        if not targets:
            continue
        if len(targets) == 1:
            g.add_edge(
                src, "calls", targets[0],
                evidence="ast", confidence=0.95, weight=1.0,
            )
        else:
            for t in targets[:5]:
                g.add_edge(
                    src, "calls", t,
                    evidence="ast", confidence=0.4, weight=1.0,
                )


def extract_all_python(
    repo: Path,
    files: list[str],
    g: Graph,
) -> dict[str, Any]:
    indexes: dict[str, Any] = {
        "fns_by_name": defaultdict(list),
        "fns_by_qual": defaultdict(list),
        "types_by_name": defaultdict(list),
        "fn_meta": {},
        "test_nodes": {},
        "py_calls": [],
        "py_types": {},
        "argparse_sites": [],
        "cli_commands": [],
    }
    module_index: dict[str, str] = {}
    for rel in files:
        module_index[rel] = f"file:{rel}"

    for rel in files:
        extract_python_file(repo, rel, g, indexes, module_index)
    resolve_python_calls(g, indexes)
    return indexes
