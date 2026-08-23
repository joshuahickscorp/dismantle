from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

IGNORED_DIRS = {".git", ".haider", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target", "coverage"}


class WorkspaceIndex:
    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self.files: List[str] = []
        self.manifests: List[str] = []
        self.test_files: List[str] = []

    def build(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), self.root)
                self.files.append(rel)
                if fn in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod", "README.md"):
                    self.manifests.append(rel)
                if fn.startswith("test_") or fn.endswith("_test.py") or fn.endswith(".test.js"):
                    self.test_files.append(rel)
        return self

    def resolve(self, text: str) -> Optional[str]:
        pattern = re.compile(r'\b([\w./-]+\.(?:md|py|js|ts|rs|go|json|toml|yml|yaml|txt))\b')
        for match in pattern.finditer(text):
            candidate = match.group(1)
            if os.path.isfile(os.path.join(self.root, candidate)):
                return candidate
        return None

    def read(self, text: str) -> Optional[Dict]:
        rel = self.resolve(text)
        if rel is None:
            return None
        try:
            content = open(os.path.join(self.root, rel)).read()
            return {"path": rel, "content": content}
        except Exception:
            return None
