"""Timing with a gate. FRONT D (G046, steer S015 §73/§74).

Every performance claim in this program goes through here, because the steer's rule
is that a claim needs exact identity, repeated samples, a baseline, and uncontended
measurement -- and because a number with a wide spread is not a measurement, as this
campaign already learned when a 286% spread nearly became a bandwidth claim.

A result is UNRELIABLE unless its interquartile spread is inside the gate, and an
UNRELIABLE arm may not win a comparison.
"""
from __future__ import annotations

from typing import Any, Callable

IQR_GATE_PCT = 10.0


def time_arm(fn: Callable[[], Any], *, reps: int = 40, warmup: int = 10) -> dict[str, Any]:
    import time
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    s.sort()
    q1, med, q3 = s[len(s) // 4], s[len(s) // 2], s[(3 * len(s)) // 4]
    iqr = (q3 - q1) / q1 * 100
    return {"median_s": med, "q1_s": q1, "q3_s": q3,
            "iqr_spread_pct": round(iqr, 2),
            "full_range_pct": round((s[-1] - s[0]) / s[0] * 100, 2),
            "reps": reps, "warmup": warmup,
            "reliable": iqr <= IQR_GATE_PCT}


def compare(arms: dict[str, dict[str, Any]], *, baseline: str,
            candidate: str) -> dict[str, Any]:
    b, c = arms[baseline], arms[candidate]
    if not (b["reliable"] and c["reliable"]):
        return {"verdict": "NO_CLAIM",
                "reason": f"an arm failed the {IQR_GATE_PCT}% IQR gate "
                          f"({baseline} {b['iqr_spread_pct']}%, "
                          f"{candidate} {c['iqr_spread_pct']}%); an unreliable arm "
                          f"may not win or lose a comparison",
                "speedup": None}
    speedup = b["median_s"] / c["median_s"]
    # the margin must clear the noise of BOTH arms, or it is not a result
    noise = max(b["iqr_spread_pct"], c["iqr_spread_pct"])
    margin = abs(speedup - 1.0) * 100
    if margin <= noise:
        return {"verdict": "INDISTINGUISHABLE", "speedup": round(speedup, 4),
                "reason": f"margin {margin:.2f}% does not clear arm noise {noise:.2f}%"}
    return {"verdict": "CANDIDATE_WINS" if speedup > 1 else "BASELINE_WINS",
            "speedup": round(speedup, 4),
            "margin_pct": round(margin, 2), "arm_noise_pct": round(noise, 2)}
