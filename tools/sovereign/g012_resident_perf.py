#!/usr/bin/env python3
"""G012 producer: a FRESH complete-token decode measurement, conditions recorded.

Receipt: receipts/sovereign/G012_resident_perf.json

Refuses to emit a partial receipt. Every exit that cannot measure honestly writes
nothing and returns non-zero with the reason.

Two things this deliberately does NOT do:

  * It does not reuse a historical 34/36/45/62 tok/s figure. `reused_historical_figure`
    is hardcoded False because the number is taken here, now, or not at all.
  * It does not report `vm.swapusage used=` as live swap. That counter is a BOOT
    HIGH-WATER MARK: it never falls, so comparing it to a ceiling latches any gate
    that reads it. The receipt records it under its real name and ALSO records the
    Swapouts delta across the measurement, which is the only figure on macOS that
    can distinguish "swapping now" from "swapped once, hours ago".
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts/sovereign/G012_resident_perf.json"
PRODUCED_BY = "tools/sovereign/g012_resident_perf.py"
SCHEMA = "hawking.sovereign.g012_resident_perf.v1"
ENVELOPE = REPO / "ascension_envelope.hawking.json"

sys.path.insert(0, str(REPO))


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERR {exc}"


def _swapouts() -> int:
    """Cumulative swapouts since boot. Only its DELTA means anything."""
    out = _sh("vm_stat")
    m = re.search(r"Swapouts:\s+(\d+)", out)
    return int(m.group(1)) if m else -1


def _free_bytes() -> int:
    out = _sh("vm_stat")
    page = re.search(r"page size of (\d+) bytes", out)
    free = re.search(r"Pages free:\s+(\d+)", out)
    if not page or not free:
        return -1
    return int(page.group(1)) * int(free.group(1))


def _contamination() -> list:
    out = _sh("ps aux | sort -k3 -rn | head -8")
    rows = []
    for line in out.splitlines()[:8]:
        parts = line.split(None, 10)
        if len(parts) >= 11:
            try:
                cpu = float(parts[2])
            except ValueError:
                continue
            rows.append({"cpu_pct": cpu, "command": parts[10][:110]})
    return rows


def _decode_rate(conn, prompt: str, max_tokens: int) -> dict:
    """One complete-token measurement: wall for a known number of decoded tokens."""
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    body = conn.complete_payload(payload, timeout=1800)
    wall = time.perf_counter() - t0
    usage = body.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    calls = body.get("model_calls") or []
    prefill_ns = 0
    if calls:
        prefill_ns = int(((calls[0].get("prefill_profile") or {}).get("totals") or {})
                         .get("wall_ns") or 0)
    return {
        "wall_s": wall,
        "completion_tokens": completion,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "prefill_wall_ns": prefill_ns,
        "decode_wall_s": max(wall - prefill_ns / 1e9, 0.0) if prefill_ns else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=128,
                    help="decoded tokens per single-stream sample")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--streams", type=int, default=2,
                    help="concurrent streams for the saturation retest")
    args = ap.parse_args()

    from hcli.hawking_native import HawkingNativeConnector, config_for_model_path

    cfg = config_for_model_path(str(ENVELOPE))
    conn = HawkingNativeConnector(cfg)

    prompt = ("Count upward from one, one number per line, and do not stop early. "
              "Write nothing else.")

    swap_before = _swapouts()
    conds_before = {
        "free_bytes": _free_bytes(),
        "background_contamination": _contamination(),
        "swap_used_bytes_boot_high_water": _sh("sysctl -n vm.swapusage"),
        "swapouts_cumulative_before": swap_before,
    }

    singles = []
    for _ in range(max(1, args.samples)):
        singles.append(_decode_rate(conn, prompt, args.tokens))

    usable = [s for s in singles if s["completion_tokens"] > 0]
    if not usable:
        print("REFUSED: no sample decoded a single token", file=sys.stderr)
        return 2

    # complete token = the whole call divided by the tokens it produced. This is
    # deliberately the COMPLETE wall, not a kernel time.
    per_token_ns = [s["wall_s"] * 1e9 / s["completion_tokens"] for s in usable]
    best = min(per_token_ns)
    median = sorted(per_token_ns)[len(per_token_ns) // 2]

    # Saturation retest: N concurrent streams against the same body.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.streams) as pool:
        futures = [pool.submit(_decode_rate, conn, prompt, args.tokens)
                   for _ in range(args.streams)]
        agg = [f.result() for f in futures]
    agg_wall = time.perf_counter() - t0
    agg_tokens = sum(a["completion_tokens"] for a in agg)

    swap_after = _swapouts()
    doc = {
        "schema": SCHEMA,
        "produced_by": PRODUCED_BY,
        "command": f"python3 {PRODUCED_BY} --tokens {args.tokens} "
                   f"--samples {args.samples} --streams {args.streams}",
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "reused_historical_figure": False,
        "resident_identity": getattr(cfg, "resident_identity", None),
        "envelope": str(ENVELOPE.relative_to(REPO)),
        "complete_token_ns": median,
        "complete_token_ns_best": best,
        "decode_tok_s": 1e9 / median,
        "decode_tok_s_best": 1e9 / best,
        "samples": singles,
        "spread_pct": (max(per_token_ns) - min(per_token_ns)) / min(per_token_ns) * 100.0,
        "aggregate_vs_single": {
            "single_tok_s": 1e9 / median,
            "aggregate_tok_s": (agg_tokens / agg_wall) if agg_wall > 0 else 0,
            "streams": args.streams,
            "aggregate_wall_s": agg_wall,
            "aggregate_completion_tokens": agg_tokens,
        },
        "conditions": {
            "free_bytes": conds_before["free_bytes"],
            "background_contamination": conds_before["background_contamination"],
            # Named for what it is. See the module docstring.
            "swap_used_bytes": conds_before["swap_used_bytes_boot_high_water"],
            "swap_used_bytes_is_boot_high_water": True,
            "swapouts_cumulative_before": swap_before,
            "swapouts_cumulative_after": swap_after,
            "swapouts_delta_during_measurement": (
                swap_after - swap_before if swap_before >= 0 and swap_after >= 0 else None
            ),
            "free_bytes_after": _free_bytes(),
        },
        "evidence": {
            "note": "complete-token wall divided by tokens actually produced; "
                    "no kernel time, no synthetic loop",
            "single_stream_samples": len(usable),
        },
    }

    for field in ("complete_token_ns", "decode_tok_s"):
        if not isinstance(doc[field], (int, float)) or doc[field] == 0:
            print(f"REFUSED: {field} is not a measurement", file=sys.stderr)
            return 3
    if not doc["aggregate_vs_single"]["aggregate_tok_s"]:
        print("REFUSED: the saturation retest produced no aggregate rate", file=sys.stderr)
        return 4

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RECEIPT}")
    print(f"  decode_tok_s   {doc['decode_tok_s']:.3f}  (best {doc['decode_tok_s_best']:.3f}, "
          f"spread {doc['spread_pct']:.1f}%)")
    a = doc["aggregate_vs_single"]
    print(f"  single {a['single_tok_s']:.3f} tok/s   aggregate {a['aggregate_tok_s']:.3f} tok/s "
          f"across {a['streams']} streams")
    print(f"  swapouts delta during measurement: "
          f"{doc['conditions']['swapouts_delta_during_measurement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
