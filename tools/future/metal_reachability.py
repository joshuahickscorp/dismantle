"""METAL REACHABILITY — falsify a host-wide blocker, or confirm it.

The whole campaign carries one sentence:

    "this host has no Metal-capable GPU"

It appears as `status: BLOCKED_NO_METAL_GPU` on the Flash teacher capture, which
is gate 2 of the meta funnel and therefore the thing standing between Flash and
every downstream representation result. It was repeated into this sidecar's own
autonomy driver as a settled fact about the machine.

It is not a fact about the machine. This host is an Apple M3 Ultra with a 60-core
GPU reporting Metal 4, and `MTLCreateSystemDefaultDevice()` returns that device
from an ordinary command-line process here. So `Device::system_default()`
returning None inside one Rust binary is a property of THAT PROCESS, not of the
host, and the two have completely different remedies: a missing GPU means buy
hardware, while a process that cannot see a present GPU means change how the
process is launched.

What this module does NOT claim:

* It does not claim the Codex capture will now succeed. It did not run that
  binary, and the failure could still be anything downstream of device creation.
* It does not measure anything. Device enumeration is a capability query --
  no queue is used, no work is submitted, no lease is taken, no timing is kept.
* It does not identify the cause of the Codex-side failure. Naming the machine
  as innocent is not the same as naming the guilty party, and guessing between
  sandbox, launch context and build target would be inventing evidence.

    python3 tools/future/metal_reachability.py --probe
    python3 tools/future/metal_reachability.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tools.future._common import REPO, write_receipt

RECEIPT = "METAL_REACHABILITY.json"
SCHEMA = "hawking.future.metal_reachability.v1"

# The claim under test, and where it is written down.
CLAIM = "this host has no Metal-capable GPU"
CLAIM_SITES = (
    "receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
    "crates/hawking-core/examples/flash_meta_teacher_trace.rs",
    "crates/hawking-core/src/metal/mod.rs",
)

# Enumeration only. No command queue is created, so no GPU work can be submitted
# by this probe even accidentally.
PROBE_SWIFT = '''import Metal
var out: [String: Any] = [:]
if let d = MTLCreateSystemDefaultDevice() {
    out["system_default"] = d.name
    out["unified_memory"] = d.hasUnifiedMemory
    out["recommended_max_working_set_bytes"] = d.recommendedMaxWorkingSetSize
} else {
    out["system_default"] = NSNull()
}
let all = MTLCopyAllDevices()
out["n_devices"] = all.count
out["devices"] = all.map { $0.name }
let data = try! JSONSerialization.data(withJSONObject: out)
print(String(data: data, encoding: .utf8)!)
'''


class ProbeUnavailable(Exception):
    pass


def probe() -> dict[str, Any]:
    """Ask Metal whether it has a device. Enumeration, not measurement."""
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise ProbeUnavailable("swiftc is not on PATH; the claim cannot be tested here")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.swift"
        binary = Path(tmp) / "probe"
        src.write_text(PROBE_SWIFT)
        build = subprocess.run([swiftc, "-O", str(src), "-o", str(binary)],
                               capture_output=True, text=True, timeout=300)
        if build.returncode != 0:
            raise ProbeUnavailable(f"probe did not build: {build.stderr.strip()[:300]}")
        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
        if run.returncode != 0:
            raise ProbeUnavailable(f"probe did not run: {run.stderr.strip()[:300]}")
        return json.loads(run.stdout.strip())


def verdict(observed: dict[str, Any] | None, why_unavailable: str = "") -> dict[str, Any]:
    if observed is None:
        return {
            "claim": CLAIM,
            "verdict": "UNTESTED",
            "why": why_unavailable or "the probe could not run on this host",
        }
    device = observed.get("system_default")
    if device:
        return {
            "claim": CLAIM,
            "verdict": "FALSIFIED_AS_A_HOST_PROPERTY",
            "why": (
                f"MTLCreateSystemDefaultDevice() returned {device!r} from an ordinary "
                f"command-line process on this host, and MTLCopyAllDevices() reports "
                f"{observed.get('n_devices')} device(s). A Rust Device::system_default() "
                f"returning None is therefore a property of that process, not of the "
                f"machine."
            ),
            "what_this_does_not_establish": [
                "that the blocked capture will now succeed; that binary was not run here",
                "the cause of the process-side failure; sandbox, launch context and "
                "build target are all consistent with what was observed and nothing "
                "here distinguishes them",
            ],
            "different_remedies": {
                "if the host had no GPU": "acquire hardware; nothing else helps",
                "a process that cannot see a present GPU": (
                    "change how the process is launched, then re-run the capture"
                ),
            },
        }
    return {
        "claim": CLAIM,
        "verdict": "CONFIRMED",
        "why": "MTLCreateSystemDefaultDevice() returned nil from an ordinary process here",
    }


def claim_sites() -> list[dict[str, Any]]:
    """Where the claim is written down, so a correction can find all of it."""
    rows = []
    for rel in CLAIM_SITES:
        path = REPO / rel
        rows.append({
            "path": rel,
            "present": path.is_file(),
            "carries_claim": bool(
                path.is_file()
                and "no Metal-capable GPU" in path.read_text(errors="replace")
            ),
        })
    return rows


def build() -> Path:
    observed: dict[str, Any] | None = None
    why = ""
    try:
        observed = probe()
    except (ProbeUnavailable, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        why = f"{type(exc).__name__}: {exc}"

    v = verdict(observed, why)
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Test the campaign-wide claim that this host has no Metal-capable GPU, "
            "because a host property and a process property have different remedies."
        ),
        "evidence_class": "STATIC_ONLY",
        "is_a_measurement": False,
        "why_not_a_measurement": (
            "device enumeration only. No command queue is created, no work is "
            "submitted, no lease is taken, and no timing is recorded."
        ),
        "gpu_authority": False,
        "observed": observed,
        "verdict": v,
        "claim_sites": claim_sites(),
        "blocked_downstream_if_true": [
            "Flash teacher capture (meta funnel gate 2: real_teacher_fit)",
            "every representation result that funnel gates",
            "physical NX qualification",
            "protected absolute measurement",
        ],
        "recovered_implementation": [
            "crates/hawking-core/src/metal/mod.rs raises Error::Metal on "
            "Device::system_default() returning None",
            "crates/hawking-core/examples/flash_meta_teacher_trace.rs writes the "
            "boundary receipt on any prefix-initialization error",
        ],
        "gaps_closed": [
            "nothing had tested the blocker; it was carried as a settled host fact",
        ],
        "negative_findings": [
            "the boundary receipt hardcodes status BLOCKED_NO_METAL_GPU for ANY "
            "prefix-initialization error, so that status is not by itself evidence "
            "that the GPU was the problem -- here the error string did name Metal",
            "this module cannot say why the Rust process saw no device",
        ],
        "resident_callable": {
            "entry_point": "tools.future.metal_reachability.probe()",
            "workunit": "one CPU_ANALYSIS unit; cheap; no GPU authority",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_EXECUTION.complete-token",
            "fails_closed": "an unavailable probe records UNTESTED, never CONFIRMED",
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/metal_reachability.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.probe:
        try:
            print(json.dumps(probe(), indent=1, sort_keys=True))
        except ProbeUnavailable as exc:
            print(json.dumps({"untested": str(exc)}, indent=1))
            return 1
        return 0
    out = build()
    print(out)
    print(json.dumps(json.loads(out.read_text())["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
