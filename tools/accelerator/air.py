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

# Memory domains an operand may be REQUIRED to live in. AIR carries the requirement;
# HUMF (G050) owns actually satisfying it. Keeping the vocabularies identical means a
# program can be handed to HUMF without translation.
MEMORY_DOMAINS = ("APPLE_UM", "MOCK_EXTERNAL_VRAM", "NVIDIA_VRAM_SIDECAR",
                  "NVIDIA_VRAM_DIRECT", "SSD_COLD", "HOST_LOGICAL", "ANY")

# Synchronization scopes. AIR can REPRESENT these; the current Metal backend can
# execute NONE of them, and lower_to_msl refuses rather than dropping them silently.
SYNC_SCOPES = ("THREADGROUP", "SIMDGROUP", "DEVICE")

# How a matmul is realised. Measured on this machine: simdgroup wins 1.42-1.62x over
# tiled at every size tried, but the two are kept as separate strategies rather than
# one being deleted, because tiled has no shape constraint while simdgroup requires
# every dimension to be a multiple of 8.
MATMUL_STRATEGIES = ("tiled", "simdgroup")


@dataclass(frozen=True)
class AirTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    memory_domain: str = "ANY"          # where this operand must live
    read_only: bool = True              # a side-effect declaration, not a hint

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}; known: {sorted(DTYPES)}")
        if self.memory_domain not in MEMORY_DOMAINS:
            raise ValueError(f"unknown memory domain {self.memory_domain!r}; "
                             f"known: {MEMORY_DOMAINS}")
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
class AirBarrier:
    """A synchronization point. AIR represents it; the Metal backend cannot yet
    execute one, and lowering REFUSES rather than dropping it -- a silently dropped
    barrier is a race, which is the worst class of bug to ship quietly."""
    scope: str

    def __post_init__(self):
        if self.scope not in SYNC_SCOPES:
            raise ValueError(f"unknown sync scope {self.scope!r}; known: {SYNC_SCOPES}")


@dataclass(frozen=True)
class AirOp:
    kind: str
    inputs: tuple[str, ...]
    output: str
    attrs: dict[str, Any] = field(default_factory=dict)
    writes_external: bool = False       # declares a side effect beyond `output`

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
    barriers: list[AirBarrier] = field(default_factory=list)
    output_domain: str = "ANY"

    def has_side_effects(self) -> bool:
        return any(o.writes_external for o in self.ops) or any(
            not t.read_only for t in self.inputs)

    def required_domains(self) -> dict[str, str]:
        d = {t.name: t.memory_domain for t in self.inputs if t.memory_domain != "ANY"}
        if self.output_domain != "ANY":
            d[self.output] = self.output_domain
        return d

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        """What AIR can REPRESENT is deliberately wider than what this backend can
        RUN. Saying which is which is the whole point."""
        if self.barriers:
            return False, (f"{len(self.barriers)} barrier(s) declared; the Metal "
                           f"backend has no synchronization lowering, and dropping a "
                           f"barrier would silently introduce a race")
        if self.has_side_effects():
            return False, ("the program declares side effects beyond its output; the "
                           "elementwise lowering assumes pure ops")
        foreign = {t.memory_domain for t in self.inputs
                   if t.memory_domain not in ("ANY", "APPLE_UM")}
        if foreign:
            return False, (f"operands require {sorted(foreign)}, which this backend "
                           f"cannot reach; HUMF must materialise them in APPLE_UM first")
        return True, "pure elementwise program over Apple unified memory"

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
    ok, why = prog.executable_on_metal_backend()
    if not ok:
        raise NotImplementedError(
            f"AIR represents this program but the Metal backend cannot execute it: "
            f"{why}. Refusing to lower rather than emitting something that ignores it.")
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


@dataclass
class AirMatmul:
    """A tiled matmul as an AIR PROGRAM, not a hand-written kernel sitting beside AIR.

    This exists because the GEMM receipt recorded exactly that gap: AIR had no tiling,
    no threadgroup allocation and only barrier REPRESENTATION rather than lowering, so
    the working kernel was MSL I wrote by hand. Now the tile size is a specialization,
    the threadgroup allocation is declared, and the barriers are EMITTED BY THE
    LOWERING -- which is what makes the THREADGROUP scope executable rather than
    refused.
    """
    name: str
    m: int
    k: int
    n: int
    tile: int = 16
    strategy: str = "tiled"          # "tiled" | "simdgroup"
    block: int = 1                   # simdgroup only: NxN grid of 8x8 tiles per simdgroup
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"
    a_domain: str = "ANY"
    b_domain: str = "ANY"

    def validate(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        if self.strategy not in MATMUL_STRATEGIES:
            raise ValueError(f"unknown matmul strategy {self.strategy!r}; "
                             f"known: {MATMUL_STRATEGIES}")
        if self.strategy == "simdgroup":
            if self.dtype != "f32":
                raise ValueError("the simdgroup strategy is f32 only here")
            if self.block not in (1, 2):
                raise ValueError(f"block {self.block} not supported; only 1 and 2 are "
                                 f"implemented")
            step = 8 * self.block
            for d, nm in ((self.m, "m"), (self.k, "k"), (self.n, "n")):
                if d % 8:
                    raise ValueError(f"simdgroup matrices are 8x8, so {nm}={d} must be "
                                     f"a multiple of 8; refusing rather than emitting a "
                                     f"kernel that would read out of bounds")
            for d, nm in ((self.m, "m"), (self.n, "n")):
                if d % step:
                    raise ValueError(f"block={self.block} makes each simdgroup cover "
                                     f"{step}x{step}, so {nm}={d} must be a multiple of "
                                     f"{step}")
        if self.tile <= 0 or self.tile & (self.tile - 1):
            raise ValueError(f"tile {self.tile} must be a positive power of two")
        if self.tile > 32:
            raise ValueError(f"tile {self.tile} exceeds 32; a {self.tile}x{self.tile} "
                             f"threadgroup would exceed Metal's 1024-thread limit")
        for d in (self.a_domain, self.b_domain):
            if d not in MEMORY_DOMAINS:
                raise ValueError(f"unknown memory domain {d!r}")

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        foreign = {d for d in (self.a_domain, self.b_domain)
                   if d not in ("ANY", "APPLE_UM")}
        if foreign:
            return False, (f"operands require {sorted(foreign)}; HUMF must materialise "
                           f"them in APPLE_UM first")
        # THREADGROUP barriers ARE executable now: this lowering emits them itself.
        return True, "tiled matmul with threadgroup tiles and emitted barriers"

    def barrier_scopes_emitted(self) -> list[str]:
        if self.strategy == "simdgroup":
            # a simdgroup executes in lockstep, so no explicit barrier is emitted --
            # the synchronization is implicit in the SIMD width
            return []
        return ["THREADGROUP", "THREADGROUP"]   # one after load, one after accumulate

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Grid and threadgroup for this strategy. They differ, so the caller must not
        guess."""
        if self.strategy == "simdgroup":
            s = 8 * self.block
            return (self.n // s * 32, self.m // s, 1), (32, 1, 1)
        return (self.n, self.m, 1), (self.tile, self.tile, 1)


def lower_matmul_to_msl(mm: AirMatmul) -> str:
    """AIR -> tiled MSL. The barriers here are EMITTED BY AIR."""
    mm.validate()
    ok, why = mm.executable_on_metal_backend()
    if not ok:
        raise NotImplementedError(f"AIR represents this matmul but the Metal backend "
                                  f"cannot execute it: {why}")
    T, ty = mm.tile, DTYPES[mm.dtype]
    if mm.strategy == "simdgroup":
        B_, K_, N_ = mm.block, mm.k, mm.n
        s = 8 * B_
        decl = "\n    ".join(
            f"simdgroup_float8x8 acc{i}{j} = make_filled_simdgroup_matrix<float,8,8>(0.0f);"
            for i in range(B_) for j in range(B_))
        loads = "\n        ".join(
            [f"simdgroup_float8x8 a{i};" for i in range(B_)] +
            [f"simdgroup_float8x8 b{j};" for j in range(B_)] +
            [f"simdgroup_load(a{i}, A + (row + {8*i}u) * {K_}u + k, {K_}u);"
             for i in range(B_)] +
            [f"simdgroup_load(b{j}, B + k * {N_}u + col + {8*j}u, {N_}u);"
             for j in range(B_)])
        macs = "\n        ".join(
            f"simdgroup_multiply_accumulate(acc{i}{j}, a{i}, b{j}, acc{i}{j});"
            for i in range(B_) for j in range(B_))
        stores = "\n    ".join(
            f"simdgroup_store(acc{i}{j}, C + (row + {8*i}u) * {N_}u + col + {8*j}u, {N_}u);"
            for i in range(B_) for j in range(B_))
        return f"""
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint row = tg_y * {s}u;
    uint col = tg_x * {s}u;
    {decl}
    for (uint k = 0; k < {K_}u; k += 8u) {{
        {loads}
        {macs}
    }}
    {stores}
"""
    return f"""
    uint gx = thread_position_in_grid.x;
    uint gy = thread_position_in_grid.y;
    uint lx = thread_position_in_threadgroup.x;
    uint ly = thread_position_in_threadgroup.y;
    threadgroup {ty} As[{T}][{T}];
    threadgroup {ty} Bs[{T}][{T}];
    float acc = 0.0f;
    for (uint k0 = 0; k0 < {mm.k}u; k0 += {T}u) {{
        As[ly][lx] = (gy < {mm.m}u && (k0 + lx) < {mm.k}u)
                   ? A[gy * {mm.k}u + k0 + lx] : ({ty})0;
        Bs[ly][lx] = ((k0 + ly) < {mm.k}u && gx < {mm.n}u)
                   ? B[(k0 + ly) * {mm.n}u + gx] : ({ty})0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < {T}u; ++k) acc += (float)As[ly][k] * (float)Bs[k][lx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (gy < {mm.m}u && gx < {mm.n}u) C[gy * {mm.n}u + gx] = ({ty})acc;
"""


def execute_matmul(mm: AirMatmul, a, b):
    """Run an AIR matmul on the Apple GPU."""
    import mlx.core as mx
    src = lower_matmul_to_msl(mm)
    hdr = "#include <metal_simdgroup_matrix>\n" if mm.strategy == "simdgroup" else ""
    kern = mx.fast.metal_kernel(name=f"air_mm_{mm.strategy}_{mm.name}",
                                input_names=["A", "B"], output_names=["C"], source=src,
                                header=hdr, ensure_row_contiguous=True)
    dt = mx.float32 if mm.dtype == "f32" else mx.float16
    grid, tg = mm.launch()
    (c,) = kern(inputs=[mx.array(a, dtype=dt), mx.array(b, dtype=dt)],
                grid=grid, threadgroup=tg,
                output_shapes=[(mm.m, mm.n)], output_dtypes=[dt])
    mx.eval(c)
    return c


REDUCE_OPS = ("sum", "max")


@dataclass
class AirSoftmax:
    """Row-wise softmax as ONE fused kernel. FRONT D / the CCL's attention prerequisite.

    The naive shape is three passes over the row -- max, then sum of exp, then divide --
    each re-reading data the previous pass already touched. This does all three inside
    one threadgroup with two barriers, so the row is read once into registers and the
    reductions happen in the SIMD units. That fusion is only expressible because
    barriers and simd reductions now lower, which is what the reduction work bought.
    """
    name: str
    rows: int
    cols: int
    threadgroup: int = 256
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.dtype != "f32":
            raise ValueError("softmax is f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of 32 "
                             f"and within Metal's 1024 limit")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")

    def barrier_scopes_emitted(self) -> list[str]:
        return ["THREADGROUP", "THREADGROUP"]     # after the max, after the sum

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "fused row softmax with two threadgroup reductions"

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (self.rows * self.threadgroup, 1, 1), (self.threadgroup, 1, 1)


def lower_softmax_to_msl(sm: AirSoftmax) -> str:
    sm.validate()
    lanes = sm.threadgroup // 32
    return f"""
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid % 32u;
    uint warp = lid / 32u;
    threadgroup float red[{lanes}];
    uint base = row * {sm.cols}u;

    // pass 1: row max, for numerical stability
    float m = -INFINITY;
    for (uint c = lid; c < {sm.cols}u; c += {sm.threadgroup}u) m = max(m, x[base + c]);
    float sm_ = simd_max(m);
    if (lane == 0u) red[warp] = sm_;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowmax = -INFINITY;
    for (uint i = 0; i < {lanes}u; ++i) rowmax = max(rowmax, red[i]);

    // pass 2: sum of exp, same threads, no re-launch
    float s = 0.0f;
    for (uint c = lid; c < {sm.cols}u; c += {sm.threadgroup}u) s += exp(x[base + c] - rowmax);
    float ss = simd_sum(s);
    if (lane == 0u) red[warp] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowsum = 0.0f;
    for (uint i = 0; i < {lanes}u; ++i) rowsum += red[i];

    // pass 3: normalise
    float inv = 1.0f / rowsum;
    for (uint c = lid; c < {sm.cols}u; c += {sm.threadgroup}u)
        out[base + c] = exp(x[base + c] - rowmax) * inv;
"""


def execute_softmax(sm: AirSoftmax, x):
    import mlx.core as mx
    src = lower_softmax_to_msl(sm)
    kern = mx.fast.metal_kernel(name=f"air_softmax_{sm.name}", input_names=["x"],
                                output_names=["out"], source=src,
                                ensure_row_contiguous=True)
    g, tg = sm.launch()
    (o,) = kern(inputs=[mx.array(x, dtype=mx.float32)], grid=g, threadgroup=tg,
                output_shapes=[(sm.rows, sm.cols)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o



@dataclass
class AirReduce:
    """A full reduction as an AIR program. FRONT D / CCL MATH.reduction_and_scan.

    Two stage rather than atomic: each threadgroup reduces its slice to one partial
    using simd_sum plus a threadgroup barrier, then the partials are reduced again.
    Atomics would collapse this to one pass but atomics are still a LARGE gap in the
    ledger, and borrowing an unbuilt capability to make this look simpler would be a
    lie about what executes.
    """
    name: str
    n: int
    op: str = "sum"
    threadgroup: int = 256
    strategy: str = "two_stage"      # "two_stage" | "atomic"
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    def validate(self) -> None:
        if self.op not in REDUCE_OPS:
            raise ValueError(f"unknown reduce op {self.op!r}; known: {REDUCE_OPS}")
        if self.dtype != "f32":
            raise ValueError("reductions are f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of the "
                             f"32-wide SIMD group and within Metal's 1024 limit")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.strategy not in ("two_stage", "atomic"):
            raise ValueError(f"unknown reduce strategy {self.strategy!r}")
        if self.strategy == "atomic" and self.op != "sum":
            raise ValueError("the atomic strategy implements sum only; Metal has no "
                             "float atomic max here, so max must use two_stage")

    def partials(self) -> int:
        return (self.n + self.threadgroup - 1) // self.threadgroup

    def barrier_scopes_emitted(self) -> list[str]:
        return ["THREADGROUP"]

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "two-stage reduction with simd_sum and one threadgroup barrier"


def lower_reduce_to_msl(rd: AirReduce) -> str:
    rd.validate()
    if rd.strategy == "atomic":
        lanes = rd.threadgroup // 32
        return f"""
    uint gid = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid % 32u;
    threadgroup float red[{lanes}];
    float v = (gid < {rd.n}u) ? x[gid] : 0.0f;
    float s = simd_sum(v);
    if (lane == 0u) red[lid / 32u] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {{
        float acc = 0.0f;
        for (uint i = 0; i < {lanes}u; ++i) acc += red[i];
        atomic_fetch_add_explicit(&out[0], acc, memory_order_relaxed);
    }}
"""
    ident = "0.0f" if rd.op == "sum" else "-INFINITY"
    simd = "simd_sum" if rd.op == "sum" else "simd_max"
    comb = "acc += partial[i];" if rd.op == "sum" else "acc = max(acc, partial[i]);"
    lanes = rd.threadgroup // 32
    return f"""
    uint gid = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    threadgroup float partial[{lanes}];
    float v = (gid < {rd.n}u) ? x[gid] : ({ident});
    float s = {simd}(v);
    uint lane = lid % 32u;
    uint warp = lid / 32u;
    if (lane == 0u) partial[warp] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {{
        float acc = {ident};
        for (uint i = 0; i < {lanes}u; ++i) {{ {comb} }}
        out[tgid] = acc;
    }}
"""


def execute_reduce(rd: AirReduce, x):
    """Stage one on the GPU; the partials are reduced again until one value remains.

    The atomic strategy does it in ONE launch instead, which is what the two-stage
    receipt named as the reason it lost to MLX by 3x.
    """
    import mlx.core as mx
    rd.validate()
    if rd.strategy == "atomic":
        kern = mx.fast.metal_kernel(name=f"air_red_atomic_{rd.name}", input_names=["x"],
                                    output_names=["out"],
                                    source=lower_reduce_to_msl(rd),
                                    ensure_row_contiguous=True, atomic_outputs=True)
        (o,) = kern(inputs=[mx.array(x, dtype=mx.float32)],
                    grid=(rd.partials() * rd.threadgroup, 1, 1),
                    threadgroup=(rd.threadgroup, 1, 1),
                    output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)
        mx.eval(o)
        return o
    src = lower_reduce_to_msl(rd)
    kern = mx.fast.metal_kernel(name=f"air_red_{rd.op}_{rd.name}", input_names=["x"],
                                output_names=["out"], source=src,
                                ensure_row_contiguous=True)
    cur = mx.array(x, dtype=mx.float32)
    n = rd.n
    while n > 1:
        step = AirReduce(rd.name, n, rd.op, rd.threadgroup)
        k2 = mx.fast.metal_kernel(name=f"air_red_{rd.op}_{n}", input_names=["x"],
                                  output_names=["out"],
                                  source=lower_reduce_to_msl(step),
                                  ensure_row_contiguous=True)
        p = step.partials()
        (cur,) = k2(inputs=[cur], grid=(p * step.threadgroup, 1, 1),
                    threadgroup=(step.threadgroup, 1, 1),
                    output_shapes=[(p,)], output_dtypes=[mx.float32])
        mx.eval(cur)
        n = p
    return cur


@dataclass
class AirAttention:
    """Single-head scaled dot-product attention as ONE fused kernel.

    scores = Q K^T / sqrt(d); P = softmax(scores); O = P V -- computed by one
    threadgroup per query row, with the score row held in threadgroup memory so it is
    never written to device memory at all. The naive shape materialises an
    (seq x seq) score matrix and a second (seq x seq) probability matrix; this writes
    neither.

    THE SCORE ROW LIVES IN THREADGROUP MEMORY, which caps seq_len. That cap is
    enforced rather than assumed: a longer sequence is REFUSED, because the honest
    answer is that this kernel does not do flash-attention's online softmax and
    therefore cannot stream an unbounded sequence.
    """
    name: str
    seq_q: int
    seq_k: int
    head_dim: int
    threadgroup: int = 256
    causal: bool = False
    dtype: str = "f32"
    device: str = "APPLE_GPU_0"

    # 32 KiB of threadgroup memory is the Metal limit; the score row plus the
    # reduction scratch must fit inside it.
    MAX_THREADGROUP_BYTES: ClassVar[int] = 32 * 1024

    def validate(self) -> None:
        if self.dtype != "f32":
            raise ValueError("attention is f32 only here")
        if self.threadgroup % 32 or not 32 <= self.threadgroup <= 1024:
            raise ValueError(f"threadgroup {self.threadgroup} must be a multiple of 32 "
                             f"and within Metal's 1024 limit")
        need = self.seq_k * 4 + (self.threadgroup // 32) * 4
        if need > self.MAX_THREADGROUP_BYTES:
            raise ValueError(
                f"seq_k={self.seq_k} needs {need} bytes of threadgroup memory, over the "
                f"{self.MAX_THREADGROUP_BYTES} limit. This kernel holds the whole score "
                f"row in threadgroup memory and does NOT implement flash-attention's "
                f"online softmax, so it cannot stream a longer sequence. Refusing "
                f"rather than silently truncating.")
        if min(self.seq_q, self.seq_k, self.head_dim) <= 0:
            raise ValueError("all dimensions must be positive")

    def barrier_scopes_emitted(self) -> list[str]:
        return ["THREADGROUP"] * 4

    def executable_on_metal_backend(self) -> tuple[bool, str]:
        return True, "fused single-head attention, score row in threadgroup memory"

    def launch(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (self.seq_q * self.threadgroup, 1, 1), (self.threadgroup, 1, 1)

    def materialised_bytes_avoided(self) -> int:
        """What the naive shape would write to device memory and this does not."""
        return 2 * self.seq_q * self.seq_k * 4


def lower_attention_to_msl(at: AirAttention) -> str:
    at.validate()
    lanes = at.threadgroup // 32
    scale = 1.0 / (at.head_dim ** 0.5)
    causal = ("if (j > q) s = -INFINITY;" if at.causal else "")
    return f"""
    uint q = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = lid % 32u;
    uint warp = lid / 32u;
    threadgroup float scores[{at.seq_k}];
    threadgroup float red[{lanes}];

    // scores = Q . K^T * scale, held in threadgroup memory and never written out
    for (uint j = lid; j < {at.seq_k}u; j += {at.threadgroup}u) {{
        float s = 0.0f;
        for (uint d = 0; d < {at.head_dim}u; ++d)
            s += Q[q * {at.head_dim}u + d] * K[j * {at.head_dim}u + d];
        s *= {scale}f;
        {causal}
        scores[j] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    for (uint j = lid; j < {at.seq_k}u; j += {at.threadgroup}u) m = max(m, scores[j]);
    float mm = simd_max(m);
    if (lane == 0u) red[warp] = mm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowmax = -INFINITY;
    for (uint i = 0; i < {lanes}u; ++i) rowmax = max(rowmax, red[i]);

    float acc = 0.0f;
    for (uint j = lid; j < {at.seq_k}u; j += {at.threadgroup}u) {{
        float e = exp(scores[j] - rowmax);
        scores[j] = e;
        acc += e;
    }}
    float ss = simd_sum(acc);
    if (lane == 0u) red[warp] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowsum = 0.0f;
    for (uint i = 0; i < {lanes}u; ++i) rowsum += red[i];
    float inv = 1.0f / rowsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O = P V, one output element per thread stride
    for (uint d = lid; d < {at.head_dim}u; d += {at.threadgroup}u) {{
        float o = 0.0f;
        for (uint j = 0; j < {at.seq_k}u; ++j) o += scores[j] * V[j * {at.head_dim}u + d];
        out[q * {at.head_dim}u + d] = o * inv;
    }}
"""


def execute_attention(at: AirAttention, q, k, v):
    import mlx.core as mx
    src = lower_attention_to_msl(at)
    kern = mx.fast.metal_kernel(name=f"air_attn_{at.name}",
                                input_names=["Q", "K", "V"], output_names=["out"],
                                source=src, ensure_row_contiguous=True)
    g, tg = at.launch()
    (o,) = kern(inputs=[mx.array(q, dtype=mx.float32), mx.array(k, dtype=mx.float32),
                        mx.array(v, dtype=mx.float32)],
                grid=g, threadgroup=tg,
                output_shapes=[(at.seq_q, at.head_dim)], output_dtypes=[mx.float32])
    mx.eval(o)
    return o


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
