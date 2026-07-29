#!/usr/bin/env python3.12
"""Prove that a retired entrypoint is still invocable through whatever replaced it.

`tools/loc/hawking_inventory.py` counts capability as entrypoints: `__main__` blocks,
argparse surfaces, binaries, scripts. Campaign section 7.2 asks for the opposite shape --
replace repeated controllers with typed specs run by one engine. So a spec-driven rewrite
fails the capability gate by construction: 77 modules with `__main__` become 77 rows in a
spec table, and the counter sees 76 capabilities vanish.

The rule (control/REBUILD_ACCOUNTING_RULES.json, `capability_equivalence_for_spec_driven_designs`)
is that a spec may replace an entrypoint only if the replacement is **invocable and proven**.
This runs that proof.

    capability_manifest.py --check
    capability_manifest.py --gate --before control/rungs/pre-s2 --after control/rungs/post-s2
    capability_manifest.py --scaffold --before control/rungs/pre-s2 --after control/rungs/post-s2

`--scaffold` writes a manifest skeleton listing every entrypoint that disappeared between
two inventory snapshots, so the lane that retired them has to fill in how each is reached
now. An entry left unfilled is a lost capability, which is exactly the reading we want.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "control" / "CAPABILITY_MANIFEST.json"
TIMEOUT = 120


def load_caps(snapshot: str) -> set[str]:
    p = Path(snapshot)
    if not p.suffix:
        p = p.with_suffix(".caps.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    return set(d.get("python_entrypoint_list", [])) | {
        b if isinstance(b, str) else b.get("name", str(b)) for b in d.get("rust_binaries", [])
    }


def run_entry(e: dict) -> dict:
    cmd = e.get("invocation")
    if not cmd:
        return {**e, "status": "fail", "detail": "no invocation recorded"}
    try:
        r = subprocess.run(
            cmd, cwd=ROOT, shell=True, capture_output=True, text=True, timeout=TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return {**e, "status": "fail", "detail": f"timed out after {TIMEOUT}s"}
    ok = r.returncode == 0
    tail = ((r.stdout or "") + (r.stderr or "")).strip()[-200:]
    return {**e, "status": "pass" if ok else "fail",
            "detail": f"exit {r.returncode}" + ("" if ok else f": {tail}")}


def check() -> dict:
    if not MANIFEST.exists():
        return {"schema": "hawking.capability_manifest_report.v1",
                "entries": [], "passed": 0, "failed": 0,
                "note": "control/CAPABILITY_MANIFEST.json does not exist; nothing claimed"}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = [run_entry(e) for e in man.get("entries", [])]
    return {
        "schema": "hawking.capability_manifest_report.v1",
        "entries": results,
        "passed": sum(1 for r in results if r["status"] == "pass"),
        "failed": sum(1 for r in results if r["status"] == "fail"),
    }


def scaffold(before: str, after: str) -> dict:
    lost = sorted(load_caps(before) - load_caps(after))
    return {
        "schema": "hawking.capability_manifest.v1",
        "note": ("Every entry below is an entrypoint that existed before this rung and does "
                 "not exist after it. Fill `invocation` with the exact command that reaches "
                 "the same capability now, or set `disposition` to \"retired\" with the "
                 "receipt that retired it. An entry left as-is is a lost capability."),
        "entries": [
            {"retired_entrypoint": p, "invocation": None, "disposition": None,
             "evidence": None}
            for p in lost
        ],
    }


def gate(before: str, after: str) -> tuple[dict, list[str]]:
    lost = load_caps(before) - load_caps(after)
    rep = check()
    claimed = {e.get("retired_entrypoint") for e in
               (json.loads(MANIFEST.read_text(encoding="utf-8")).get("entries", [])
                if MANIFEST.exists() else [])}
    bad: list[str] = []
    unaccounted = sorted(lost - claimed)
    if unaccounted:
        bad.append(f"{len(unaccounted)} retired entrypoints are not in the manifest at all: "
                   f"{unaccounted[:6]}{' …' if len(unaccounted) > 6 else ''}")
    for e in rep["entries"]:
        if e["status"] != "pass" and not (e.get("disposition") == "retired" and e.get("evidence")):
            bad.append(f"not invocable: {e.get('retired_entrypoint')} -- {e['detail'][:120]}")
    rep["lost_entrypoints"] = len(lost)
    rep["unaccounted"] = unaccounted
    return rep, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--scaffold", action="store_true")
    ap.add_argument("--before"); ap.add_argument("--after")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.scaffold:
        if not (args.before and args.after):
            print("--scaffold needs --before and --after", file=sys.stderr); return 2
        doc = scaffold(args.before, args.after)
        out = Path(args.out) if args.out else MANIFEST
        out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"{len(doc['entries'])} retired entrypoints -> {out}")
        return 0

    if args.gate:
        if not (args.before and args.after):
            print("--gate needs --before and --after", file=sys.stderr); return 2
        rep, bad = gate(args.before, args.after)
        print(f"manifest: {rep['passed']} invocable, {rep['failed']} not; "
              f"{rep['lost_entrypoints']} entrypoints retired this rung")
        for b in bad:
            print(f"  ! {b}")
        if args.out:
            Path(args.out).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        return 1 if bad else 0

    rep = check()
    print(f"{rep['passed']} invocable, {rep['failed']} not")
    for e in rep["entries"]:
        if e["status"] != "pass":
            print(f"  ! {e.get('retired_entrypoint')}: {e['detail'][:140]}")
    return 1 if rep["failed"] else 0


def _selfcheck() -> None:
    """Smallest thing that fails if the accounting logic breaks."""
    e_ok = run_entry({"retired_entrypoint": "x", "invocation": "true"})
    e_no = run_entry({"retired_entrypoint": "y", "invocation": "false"})
    assert e_ok["status"] == "pass" and e_no["status"] == "fail", (e_ok, e_no)
    assert run_entry({"retired_entrypoint": "z"})["status"] == "fail", "missing invocation must fail"
    # an entrypoint that vanished and is not claimed anywhere must be reported unaccounted
    lost, claimed = {"a", "b"}, {"a"}
    assert sorted(lost - claimed) == ["b"]
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
