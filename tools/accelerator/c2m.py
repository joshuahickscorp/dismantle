"""C2M — the CUDA to Metal frontend. FRONT C (G045, steer S015).

This lowers a RESTRICTED subset of CUDA C into AIR, which then executes on the
Apple GPU. The subset is small and its boundary is enforced: anything outside it is
REFUSED with the exact construct named, because the steer requires unsupported
paths to fail explicitly rather than silently producing something plausible.

CONFORMANCE HONESTY. This is C2M-T0 -- trivial elementwise kernels -- and it may
not be described as "CUDA support" without that tier. It is also NOT a CUDA
differential: no NVIDIA hardware exists on this machine, so a translated kernel is
graded against a numpy oracle running on the CPU, not against CUDA. That is a
weaker claim and it is labelled as one everywhere it appears. P2 (a real CUDA
differential corpus) stays open and cannot be closed here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from air import AirOp, AirProgram, AirTensor

# Constructs we can see but deliberately do not handle. Naming them is the point:
# a silent mistranslation is worse than a refusal.
UNSUPPORTED = {
    "__shared__": "shared memory",
    "__syncthreads": "block synchronization",
    "atomicAdd": "atomics",
    "atomicCAS": "atomics",
    "for": "loops",
    "while": "loops",
    "__ldg": "read-only cache intrinsics",
    "cooperative_groups": "cooperative groups",
    "printf": "device printf",
    "double": "fp64",
    "__syncwarp": "warp synchronization",
    "__shfl": "warp shuffle",
}

# The T0 expression forms. A pattern frontend, not a general C expression parser --
# stated plainly so nobody mistakes its reach.
PATTERNS: list[tuple[str, str, int]] = [
    (r"^(\w+)\[i\]\s*\+\s*(\w+)\[i\]$", "add", 2),
    (r"^(\w+)\[i\]\s*\*\s*(\w+)\[i\]$", "mul", 2),
    (r"^(\w+)\[i\]\s*-\s*(\w+)\[i\]$", "sub", 2),
    (r"^fmaxf\(\s*(\w+)\[i\]\s*,\s*0(?:\.0)?f?\s*\)$", "relu", 1),
    (r"^fmax\(\s*(\w+)\[i\]\s*,\s*0(?:\.0)?f?\s*\)$", "relu", 1),
]

GRID_IDX = re.compile(
    r"blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x"
    r"|threadIdx\.x\s*\+\s*blockIdx\.x\s*\*\s*blockDim\.x")


class C2MRefusal(Exception):
    """Raised when a kernel falls outside the supported subset. Carries the exact
    construct, never a vague 'unsupported kernel'."""


@dataclass
class TranslatedKernel:
    name: str
    params: list[tuple[str, str, bool]]     # (ctype, name, is_pointer)
    inputs: list[str]
    output: str
    program: AirProgram
    tier: str = "C2M-T0"


def _check_unsupported(src: str) -> None:
    for token, human in UNSUPPORTED.items():
        if re.search(r"\b" + re.escape(token), src):
            raise C2MRefusal(
                f"{human} is not in the C2M-T0 subset (found {token!r}); "
                f"refusing rather than mistranslating")


def translate(src: str, *, elements: int) -> TranslatedKernel:
    _check_unsupported(src)

    m = re.search(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{(.*)\}", src, re.S)
    if not m:
        raise C2MRefusal("no __global__ kernel with a body was found")
    name, params_src, body = m.group(1), m.group(2), m.group(3)

    params: list[tuple[str, str, bool]] = []
    for raw in [p.strip() for p in params_src.split(",") if p.strip()]:
        pm = re.match(r"(?:const\s+)?(\w+)\s*(\*?)\s*(\w+)$", raw)
        if not pm:
            raise C2MRefusal(f"cannot parse parameter {raw!r}")
        ctype, star, pname = pm.group(1), pm.group(2), pm.group(3)
        if ctype not in ("float", "int"):
            raise C2MRefusal(f"parameter type {ctype!r} is outside the subset")
        params.append((ctype, pname, star == "*"))

    if not GRID_IDX.search(body):
        raise C2MRefusal("no recognised global thread index "
                         "(expected blockIdx.x * blockDim.x + threadIdx.x)")

    # the single guarded assignment this tier supports
    am = re.search(r"(\w+)\s*\[\s*i\s*\]\s*=\s*([^;]+);", body)
    if not am:
        raise C2MRefusal("no single indexed assignment found in the kernel body")
    out_name, expr = am.group(1), am.group(2).strip()

    if len(re.findall(r"\w+\s*\[\s*i\s*\]\s*=", body)) > 1:
        raise C2MRefusal("more than one store; the T0 subset is a single assignment")

    for pat, op, arity in PATTERNS:
        pm = re.match(pat, expr)
        if not pm:
            continue
        srcs = list(pm.groups())[:arity]
        ptr_names = [p[1] for p in params if p[2]]
        for s in srcs:
            if s not in ptr_names:
                raise C2MRefusal(f"{s!r} is not a pointer parameter")
        ins = [AirTensor(s, (elements,), "f32") for s in srcs]
        prog = AirProgram(name, ins, [AirOp(op, tuple(srcs), "out_")], "out_")
        return TranslatedKernel(name, params, srcs, out_name, prog)

    raise C2MRefusal(f"expression {expr!r} is outside the C2M-T0 pattern set "
                     f"(supported: a+b, a*b, a-b, fmaxf(a,0))")


def conformance(results: list[dict[str, Any]]) -> dict[str, Any]:
    """A tier is claimed only from what actually executed and matched."""
    passed = [r for r in results if r.get("matches_oracle")]
    return {
        "tier_claimed": "C2M-T0" if passed else "NONE",
        "tier_definition": "T0 = trivial elementwise kernels",
        "kernels_translated": len(results),
        "kernels_matching_oracle": len(passed),
        "higher_tiers": {
            "C2M-T1": "NOT CLAIMED: no runtime, memory API or stream semantics exist",
            "C2M-T2": "NOT CLAIMED: no math/ML kernel corpus (no GEMM, no reduction)",
            "C2M-T3": "NOT CLAIMED: no real open CUDA project has been run",
            "C2M-T4": "NOT CLAIMED: no AI workload",
            "C2M-T5": "NOT CLAIMED: nothing is production-supported",
        },
        "oracle": "numpy on CPU",
        "is_a_cuda_differential": False,
        "why_not": "no NVIDIA hardware is present, so nothing was compared against "
                   "CUDA itself; P2 remains open",
    }
