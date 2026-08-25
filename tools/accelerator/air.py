"""AIR — the Accelerator IR, and its Metal lowering. FRONT A (G043, steer S015).

AIR is Hawking's internal representation of GPU computation. This is the minimum
that is REAL rather than described: a program is built, lowered to MSL, executed on
the Apple GPU, and graded against a numpy oracle. The steer's proof P3 is exactly
that -- AIR executes a basic operation on Metal -- so nothing here is allowed to be
a schema with no execution behind it.

What AIR represents today: operations, operands with dtype and shape, dependencies
(via SSA names), device requirements, and specialization constants. What it does
NOT yet represent, stated so the gap is not mistaken for coverage: explicit
synchronization primitives, side effects, multi-device placement constraints, and
memory-domain requirements. Those arrive with HUMF (G050) and EGB (G051/G052); AIR
must not need redesign when they do, which is why placement is already a field on
the program rather than an afterthought.

Execution goes through mx.fast.metal_kernel because `xcrun metal` is not present on
this machine (no Metal developer toolchain), so ahead-of-time metallib compilation
is unavailable here. MLX JIT-compiles the MSL we emit. That is a real dependency
and it is recorded as one, not hidden.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any

DTYPES = {"f32": "float", "f16": "half"}


@dataclass(frozen=True)
class AirTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}; known: {sorted(DTYPES)}")
        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError(f"bad shape {self.shape!r}")

    @property
    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


# Each op lowers to one elementwise MSL expression over index `i`. Deliberately
# narrow: this is the P3 corpus, not the whole math library the steer names.
# Operands arrive already resolved: a buffer reads as `name[i]`, an SSA
# intermediate as the scalar `name`. The templates must therefore never write
# `[i]` themselves -- doing so indexed a scalar and failed to compile.
ELEMENTWISE = {
    "add": "{a} + {b}",
    "mul": "{a} * {b}",
    "sub": "{a} - {b}",
    "saxpy": "ALPHA * {a} + {b}",
    "relu": "max(({t}){a}, ({t})0)",
    "silu": "{a} / ({t})(1.0 + exp(-(float){a}))",
}


@dataclass(frozen=True)
class AirOp:
    kind: str
    inputs: tuple[str, ...]
    output: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ELEMENTWISE:
            raise ValueError(f"unknown AIR op {self.kind!r}; known: {sorted(ELEMENTWISE)}")
        want = 1 if self.kind in ("relu", "silu") else 2
        if len(self.inputs) != want:
            raise ValueError(f"{self.kind} takes {want} input(s), got {len(self.inputs)}")


@dataclass
class AirProgram:
    """A program plus its device requirement. Placement is a field from the start so
    that EGB does not later force an AIR redesign (steer: 'EGB must not force AIR
    redesign')."""
    name: str
    inputs: list[AirTensor]
    ops: list[AirOp]
    output: str
    device: str = "APPLE_GPU_0"
    specialization: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        live = {t.name for t in self.inputs}
        for op in self.ops:
            missing = [n for n in op.inputs if n not in live]
            if missing:
                raise ValueError(f"op {op.kind} reads undefined {missing}")
            if op.output in live:
                raise ValueError(f"{op.output} assigned twice; AIR is SSA")
            live.add(op.output)
        if self.output not in live:
            raise ValueError(f"program output {self.output!r} is never produced")
        shapes = {t.shape for t in self.inputs}
        if len(shapes) != 1:
            raise ValueError(f"elementwise AIR needs one shape, got {shapes}")
        dts = {t.dtype for t in self.inputs}
        if len(dts) != 1:
            raise ValueError(f"mixed dtypes not supported yet: {dts}")

    @property
    def dtype(self) -> str:
        return self.inputs[0].dtype


def lower_to_msl(prog: AirProgram) -> str:
    """AIR -> MSL body. MLX supplies the kernel signature and thread position."""
    prog.validate()
    t = DTYPES[prog.dtype]
    lines = ["uint i = thread_position_in_grid.x;",
             f"if (i >= {prog.inputs[0].elements}u) return;"]
    for k, v in prog.specialization.items():
        lines.append(f"const float {k} = {float(v)!r}f;")
    buffers = {x.name for x in prog.inputs}          # indexed with [i]
    scalars: set[str] = set()                        # SSA intermediates
    for op in prog.ops:
        def ref(nm: str) -> str:
            return f"{nm}[i]" if nm in buffers else nm
        rhs = ELEMENTWISE[op.kind].format(
            a=ref(op.inputs[0]),
            b=ref(op.inputs[1]) if len(op.inputs) > 1 else "",
            t=t)
        if op.output == prog.output:
            # only the final op writes the buffer MLX allocated for us
            lines.append(f"out[i] = ({t})({rhs});")
        else:
            lines.append(f"{t} {op.output} = ({t})({rhs});")
            scalars.add(op.output)
    return "\n    ".join(lines)


def machine_identity() -> dict[str, Any]:
    """MachineIdentity for the receipt. The steer: no result without physical
    identity."""
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {
        "soc": sysctl("machdep.cpu.brand_string"),
        "cpu_cores": sysctl("hw.ncpu"),
        "memory_bytes": sysctl("hw.memsize"),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
    }


def execute(prog: AirProgram, arrays: dict[str, Any]):
    """Run the lowered program on the Apple GPU via MLX's JIT."""
    import mlx.core as mx
    prog.validate()
    src = lower_to_msl(prog)
    names = [t.name for t in prog.inputs]
    kern = mx.fast.metal_kernel(
        name=f"air_{prog.name}", input_names=names, output_names=["out"], source=src)
    n = prog.inputs[0].elements
    mxdt = mx.float32 if prog.dtype == "f32" else mx.float16
    (out,) = kern(
        inputs=[mx.array(arrays[k], dtype=mxdt) for k in names],
        grid=(n, 1, 1), threadgroup=(min(256, n), 1, 1),
        output_shapes=[prog.inputs[0].shape], output_dtypes=[mxdt])
    mx.eval(out)
    return out
