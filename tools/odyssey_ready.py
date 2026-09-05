#!/usr/bin/env python3
"""Odyssey I required graph: can the machine start the first campaign?

Each capability is READY only when a named module exists AND has a live caller
or entrypoint. Existence alone is not readiness -- this campaign has already
found four capabilities that were built, declared and structurally unreachable.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: capability -> (module that must exist, symbol that must be defined)
#: Paths were located by search, not guessed. A guessed path reports MISSING for
#: a capability the machine has, which is the same false negative that made the
#: first classifier call live code dead.
REQUIRED = {
    "enumerate_specimens":    ("tools/odyssey/inventory.py", None),
    "identify_architecture":  ("tools/odyssey/arch_recognizer.py", "recognize"),
    "inspect_tensors":        ("tools/headless/noetic_organ_census.py", None),
    "stream_from_modellake":  ("hcli/agentos/modellake_supervisor.py", None),
    "cheap_structural_probe": ("tools/odyssey/specimen_open.py", None),
    "gravity_experiments":    ("tools/gravity_verify_source.py", None),
    "capability_tests":       ("tools/odyssey/performance_qualification.py", None),
    "physical_benchmark":     ("tools/odyssey/runtime_authority.py", None),
    "persist_candidates":     ("tools/odyssey/contracts.py", None),
    "emit_receipts":          ("hcli/agentos/modellake_receipts.py", None),
    "derive_laws_scars":      ("tools/future/campaign_scars.py", None),
    "compare_candidates":     ("tools/odyssey/tournament.py", None),
    "schedule_followups":     ("tools/future/frontiers.py", None),
}


@lru_cache(maxsize=1)
def _index():
    """The repo-wide import/subprocess index. The engine is
    tools/future/capability_reachability.py -- this file does not write a second one."""
    sys.path.insert(0, str(REPO))
    from tools.future import capability_reachability as cr
    with cr.using_source("worktree"):
        return cr, cr.build_repo_index(source="worktree")


@lru_cache(maxsize=None)
def callers(module: str) -> tuple[int, int]:
    """(production, test) real call sites for a module: python imports of it plus
    argv/subprocess launches of its path, excluding the module's own file.

    The predecessor counted `git grep -l --fixed-strings <stem>` hits, which is
    every file containing the WORD -- a roadmap catalog naming the path as data,
    a doc, the module's own usage docstring. It reported 257 "callers" for
    inventory (2 production import sites) and 240 for tournament (1). A readiness
    gate that counts word occurrences cannot fail, so its verdict was vacuous.
    """
    cr, idx = _index()
    p = REPO / module
    with cr.using_source("worktree"):
        sites = cr.find_module_import_sites(idx, cr.module_name_of(p), exclude_files=[p])
        sites += cr._subprocess_path_sites(module, idx.files, exclude_files=[p])
        prod, test = cr._partition(sites)
    return len(prod), len(test)


def classify(module: str, symbol: str | None) -> tuple[str, str]:
    p = REPO / module
    if not p.is_file():
        alt = list((REPO / Path(module).parent).glob(f"*{Path(module).stem.split('_')[0]}*.py")) \
            if (REPO / Path(module).parent).is_dir() else []
        if alt:
            return "PARTIAL", f"absent; nearest present: {alt[0].relative_to(REPO)}"
        return "MISSING", "module absent"
    if symbol:
        try:
            if symbol not in p.read_text(encoding="utf-8", errors="replace"):
                return "PARTIAL", f"present but {symbol}() not defined"
        except OSError:
            return "PARTIAL", "unreadable"
    prod, test = callers(module)
    if prod == 0:
        # Test-only wiring is not production wiring. Reported, never counted.
        return "PARTIAL", (f"present but reachable only from {test} test call site(s)"
                           if test else "present but no call site")
    return "READY", f"{prod} production call sites" + (f" (+{test} test)" if test else "")


def main() -> int:
    rows = {k: classify(m, s) for k, (m, s) in REQUIRED.items()}
    counts: dict[str, int] = {}
    for state, _ in rows.values():
        counts[state] = counts.get(state, 0) + 1
    width = max(len(k) for k in rows)
    for k, (state, why) in rows.items():
        print(f"  {k:<{width}}  {state:8}  {why}")
    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    (REPO / ".hcli" / "odyssey_ready.json").write_text(
        json.dumps({k: {"state": v[0], "why": v[1], "module": REQUIRED[k][0],
                        "production_call_sites": callers(REQUIRED[k][0])[0],
                        "test_call_sites": callers(REQUIRED[k][0])[1]}
                    for k, v in rows.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
