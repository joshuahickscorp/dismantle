#!/usr/bin/env python3
"""Fill the Odyssey model lake in parallel, under a hard capacity guard.

Directive S011 §43-§48 and §83-§87, steer S012: keep downloading, stay ahead of the GPU,
never exhaust the drive. The whole recovered queue is ~3057 GiB against a 3.5 TB
allocation, so the queue itself is the fill plan.

Ordering is diversity-first, not biggest-first. Four workers take the remaining specimens
smallest-first so architecture coverage arrives early and the GPU is never waiting on
basic archaeology; one dedicated bulk worker streams the giant in parallel so bytes keep
landing the whole time.

Throughput is recorded per worker so the equilibrium in §84 is MEASURED rather than
assumed.
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/odyssey"))
import modellake as ml  # noqa: E402

STATE = ml.LAKE / "filler-state.json"
LOGDIR = ml.LAKE / "filler-logs"
_lock = threading.Lock()
_results = []


def plan():
    sel = json.load(open(REPO / "receipts/headless/MODEL_2_SELECTION.json"))
    resident = ml.resident_slugs()
    todo = []
    for c in sel["candidates"]:
        s = c.get("score") or {}
        if not s.get("acquirable") or not s.get("download_gib"):
            continue
        slug = c["canonical_source"].replace("/", "--") + "@" + c["canonical_revision"][:12]
        if slug in resident:
            continue
        todo.append({"oxx": c["oxx"], "repo": c["canonical_source"],
                     "revision": c["canonical_revision"], "gib": s["download_gib"],
                     "slug": slug, "novelty_axes": s.get("novelty_axes", []),
                     "model_type": s.get("model_type")})
    todo.sort(key=lambda t: t["gib"])
    return todo


def capacity_ok(gib):
    """Recompute physical free space every time. §83: never trust an old number."""
    want = int(gib * 2**30)
    ok, why = ml.admit(want, tier=2)
    free = ml.free("/Volumes/corpdrive")
    # keep real headroom on the volume beyond the lake's own budget
    headroom_ok = (free - want) > 300 * 2**30
    return (ok and headroom_ok), {
        "budget_ok": ok, "budget_reason": why,
        "free_bytes": free, "free_gib": round(free / 2**30, 1),
        "headroom_after_gib": round((free - want) / 2**30, 1),
        "headroom_ok": headroom_ok}


def fetch(item, worker):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    ok, cap = capacity_ok(item["gib"])
    if not ok:
        rec = {**item, "worker": worker, "acquired": False, "refused_capacity": cap}
        with _lock:
            _results.append(rec)
        print(f"  [w{worker}] REFUSED {item['oxx']} {item['repo']}: {cap}", flush=True)
        return rec
    t0 = time.time()
    print(f"  [w{worker}] START  {item['oxx']} {item['repo']} ({item['gib']} GiB)", flush=True)
    log = LOGDIR / f"{item['slug']}.log"
    with open(log, "w") as fh:
        p = subprocess.run(
            [sys.executable, str(REPO / "tools/odyssey/modellake.py"), "acquire",
             "--repo", item["repo"], "--revision", item["revision"]],
            stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
    wall = time.time() - t0
    got = ml.du(ml.TIER2 / item["slug"])
    rec = {**item, "worker": worker, "exit_code": p.returncode,
           "acquired": got > 0, "bytes_on_disk": got,
           "wall_s": round(wall, 1),
           "MB_per_s": round(got / 2**20 / wall, 2) if wall > 0 else None,
           "capacity_check": cap, "log": str(log)}
    with _lock:
        _results.append(rec)
        STATE.write_text(json.dumps({"results": _results}, indent=1))
    print(f"  [w{worker}] {'DONE  ' if rec['acquired'] else 'FAIL  '} {item['oxx']} "
          f"{rec['MB_per_s']} MB/s in {rec['wall_s']}s", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="workers on the diversity queue (smallest first)")
    ap.add_argument("--bulk-workers", type=int, default=1,
                    help="dedicated workers streaming the largest specimens")
    ap.add_argument("--emit", required=True)
    a = ap.parse_args()

    todo = plan()
    if not todo:
        print("nothing to acquire: every acquirable specimen is already resident")
        Path(a.emit).write_text(json.dumps({"nothing_to_do": True}, indent=1))
        return 0

    bulk = todo[-a.bulk_workers:] if a.bulk_workers else []
    small = todo[:len(todo) - len(bulk)]
    print(f"plan: {len(todo)} specimens, {sum(t['gib'] for t in todo):.1f} GiB total")
    print(f"  diversity queue ({a.workers} workers): "
          f"{[t['oxx'] for t in small]}")
    print(f"  bulk stream ({len(bulk)} workers): {[t['oxx'] for t in bulk]}")

    t0 = time.time()
    q = list(small)
    qlock = threading.Lock()

    def diversity_worker(wid):
        while True:
            with qlock:
                if not q:
                    return
                item = q.pop(0)
            fetch(item, wid)

    threads = [threading.Thread(target=diversity_worker, args=(i + 1,), daemon=False)
               for i in range(a.workers)]
    for i, item in enumerate(bulk):
        threads.append(threading.Thread(target=fetch, args=(item, f"bulk{i+1}"),
                                        daemon=False))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall = time.time() - t0
    got = [r for r in _results if r.get("acquired")]
    total_bytes = sum(r.get("bytes_on_disk", 0) for r in got)
    out = {
        "schema": "hawking.headless.lake_filler.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/lake_filler.py",
        "obligation": "G041 — ODYSSEY_ACQUISITION_CONTINUUM (S011 §43-§48, §83-§87; "
                      "steer S012: fill the allocation)",
        "hand_authored": False,
        "workers": {"diversity": a.workers, "bulk": a.bulk_workers},
        "n_planned": len(todo), "n_acquired": len(got),
        "n_refused_capacity": sum(1 for r in _results if r.get("refused_capacity")),
        "bytes_acquired": total_bytes,
        "gib_acquired": round(total_bytes / 2**30, 1),
        "wall_s": round(wall, 1),
        "aggregate_MB_per_s": round(total_bytes / 2**20 / wall, 2) if wall else None,
        "per_specimen": _results,
        "throughput_by_worker": {},
        "lake_after": {"used_bytes": ml.du(ml.LAKE),
                       "used_gib": round(ml.du(ml.LAKE) / 2**30, 1),
                       "budget_bytes": ml.TIER2_BUDGET,
                       "free_volume_gib": round(ml.free("/Volumes/corpdrive") / 2**30, 1),
                       "resident": sorted(ml.resident_slugs())},
        "pass": bool(got),
    }
    by = {}
    for r in got:
        by.setdefault(str(r["worker"]), []).append(r.get("MB_per_s") or 0)
    out["throughput_by_worker"] = {k: {"n": len(v), "mean_MB_per_s": round(sum(v) / len(v), 2)}
                                   for k, v in by.items()}
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"\nacquired {len(got)}/{len(todo)}  {out['gib_acquired']} GiB  "
          f"aggregate {out['aggregate_MB_per_s']} MB/s")
    print(f"lake now {out['lake_after']['used_gib']} GiB, volume free "
          f"{out['lake_after']['free_volume_gib']} GiB")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
