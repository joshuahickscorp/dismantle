"""FRONT A pins. Execution tests need MLX; schema tests do not."""
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/accelerator"))
import air, receipt  # noqa: E402

np = pytest.importorskip("numpy")


def test_dtype_and_shape_are_validated():
    with pytest.raises(ValueError):
        air.AirTensor("x", (4,), dtype="int4")
    with pytest.raises(ValueError):
        air.AirTensor("x", (0,))


def test_air_is_ssa():
    p = air.AirProgram("dup", [air.AirTensor("x", (4,)), air.AirTensor("y", (4,))],
                       [air.AirOp("add", ("x", "y"), "z"),
                        air.AirOp("mul", ("x", "y"), "z")], "z")
    with pytest.raises(ValueError, match="SSA"):
        p.validate()


def test_reading_an_undefined_operand_is_refused():
    p = air.AirProgram("bad", [air.AirTensor("x", (4,))],
                       [air.AirOp("relu", ("nope",), "z")], "z")
    with pytest.raises(ValueError, match="undefined"):
        p.validate()


def test_unproduced_output_is_refused():
    p = air.AirProgram("bad", [air.AirTensor("x", (4,))],
                       [air.AirOp("relu", ("x",), "t")], "z")
    with pytest.raises(ValueError, match="never produced"):
        p.validate()


def test_intermediates_lower_as_scalars_not_buffers():
    """The bug that actually happened: an SSA intermediate was declared a scalar and
    then indexed `t[i]`, which does not compile."""
    p = air.AirProgram("f", [air.AirTensor("x", (8,)), air.AirTensor("y", (8,))],
                       [air.AirOp("saxpy", ("x", "y"), "t"),
                        air.AirOp("relu", ("t",), "z")], "z",
                       specialization={"ALPHA": 2.0})
    msl = air.lower_to_msl(p)
    # `"t[i]" not in msl` is the wrong check: it matches inside `out[i]`.
    assert not re.search(r"\bt\[i\]", msl)
    assert "float t = " in msl
    assert "out[i] = " in msl
    assert "const float ALPHA" in msl


def test_only_the_final_op_writes_the_output_buffer():
    p = air.AirProgram("f", [air.AirTensor("x", (8,)), air.AirTensor("y", (8,))],
                       [air.AirOp("mul", ("x", "y"), "a"),
                        air.AirOp("relu", ("a",), "z")], "z")
    assert air.lower_to_msl(p).count("out[i] = ") == 1


def test_placement_is_a_field_so_egb_cannot_force_a_redesign():
    p = air.AirProgram("f", [air.AirTensor("x", (4,))],
                       [air.AirOp("relu", ("x",), "z")], "z", device="NVIDIA_GPU_0")
    assert p.device == "NVIDIA_GPU_0"


def test_receipt_refuses_a_missing_identity():
    with pytest.raises(ValueError, match="missing identities"):
        receipt.build(experiment_class="ACCEL-KERNEL", knowledge_level="INSTANCE",
                      identities={"machine": {}}, result={}, claim_boundary="x",
                      passed=True)


def test_receipt_refuses_absent_without_a_reason():
    ids = {k: {"status": "ABSENT", "reason": "n/a"} for k in receipt.IDENTITIES}
    ids["transport"] = {"status": "ABSENT"}
    with pytest.raises(ValueError, match="without a reason"):
        receipt.build(experiment_class="ACCEL-KERNEL", knowledge_level="INSTANCE",
                      identities=ids, result={}, claim_boundary="x", passed=True)


def test_receipt_refuses_an_invented_experiment_class():
    ids = {k: receipt.absent("n/a") for k in receipt.IDENTITIES}
    with pytest.raises(ValueError, match="not a canonical class"):
        receipt.build(experiment_class="ACCEL-VIBES", knowledge_level="INSTANCE",
                      identities=ids, result={}, claim_boundary="x", passed=True)


def test_receipt_refuses_promotion_to_an_unknown_level():
    ids = {k: receipt.absent("n/a") for k in receipt.IDENTITIES}
    with pytest.raises(ValueError, match="not a knowledge level"):
        receipt.build(experiment_class="ACCEL-KERNEL", knowledge_level="UNIVERSAL",
                      identities=ids, result={}, claim_boundary="x", passed=True)




def test_metal_execution_matches_numpy():
    pytest.importorskip("mlx.core")
    n = 1024
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    p = air.AirProgram("t", [air.AirTensor("x", (n,)), air.AirTensor("y", (n,))],
                       [air.AirOp("saxpy", ("x", "y"), "t"),
                        air.AirOp("relu", ("t",), "z")], "z",
                       specialization={"ALPHA": 2.5})
    got = np.array(air.execute(p, {"x": x, "y": y}))
    assert np.abs(got - np.maximum(2.5 * x + y, 0)).max() < 1e-5


# --- synchronization, memory domains, side effects (the gap G043 named) ---

def test_air_represents_a_barrier_and_refuses_to_lower_it():
    """A silently dropped barrier is a race. Refusing beats emitting."""
    p = air.AirProgram("b", [air.AirTensor("x", (8,))],
                       [air.AirOp("relu", ("x",), "z")], "z",
                       barriers=[air.AirBarrier("THREADGROUP")])
    ok, why = p.executable_on_metal_backend()
    assert not ok and "barrier" in why
    with pytest.raises(NotImplementedError, match="barrier"):
        air.lower_to_msl(p)


def test_an_unknown_sync_scope_is_refused():
    with pytest.raises(ValueError, match="unknown sync scope"):
        air.AirBarrier("GLOBAL_ISH")


def test_side_effects_are_declared_and_block_lowering():
    p = air.AirProgram("s", [air.AirTensor("x", (8,))],
                       [air.AirOp("relu", ("x",), "z", writes_external=True)], "z")
    assert p.has_side_effects()
    with pytest.raises(NotImplementedError, match="side effects"):
        air.lower_to_msl(p)


def test_a_writable_operand_counts_as_a_side_effect():
    p = air.AirProgram("s", [air.AirTensor("x", (8,), read_only=False)],
                       [air.AirOp("relu", ("x",), "z")], "z")
    assert p.has_side_effects()


def test_an_unknown_memory_domain_is_refused():
    with pytest.raises(ValueError, match="unknown memory domain"):
        air.AirTensor("x", (8,), memory_domain="SOMEWHERE")


def test_a_foreign_domain_blocks_lowering_rather_than_being_ignored():
    p = air.AirProgram("v", [air.AirTensor("x", (8,), memory_domain="NVIDIA_VRAM_DIRECT")],
                       [air.AirOp("relu", ("x",), "z")], "z")
    with pytest.raises(NotImplementedError, match="HUMF"):
        air.lower_to_msl(p)


def test_apple_um_is_not_treated_as_foreign():
    p = air.AirProgram("v", [air.AirTensor("x", (8,), memory_domain="APPLE_UM")],
                       [air.AirOp("relu", ("x",), "z")], "z")
    assert p.executable_on_metal_backend()[0]


def test_air_domain_vocabulary_hands_to_humf_without_translation():
    """The point of sharing the vocabulary: an AIR requirement is directly a HUMF
    domain name, so no mapping layer can drift between them."""
    import humf
    p = air.AirProgram("v", [air.AirTensor("w", (8,), memory_domain="MOCK_EXTERNAL_VRAM")],
                       [air.AirOp("relu", ("w",), "z")], "z")
    req = p.required_domains()
    assert req == {"w": "MOCK_EXTERNAL_VRAM"}
    mp = humf.MockExternalMemoryProvider(capacity_bytes=1 << 20, bandwidth_gb_s=5.0)
    # the name AIR emitted is the name HUMF uses, with no translation in between
    assert req["w"] == mp.domain.name
    fabric = humf.Humf({"APPLE_UM": humf.Domain("APPLE_UM", 1 << 30, 589.73, physical=True),
                        mp.domain.name: mp.domain})
    assert req["w"] in fabric.domains


# --- AIR produces the tiled matmul rather than sitting beside a hand-written one ---

def test_air_matmul_lowering_emits_tiles_and_barriers():
    mm = air.AirMatmul("m", 64, 64, 64, tile=16)
    src = air.lower_matmul_to_msl(mm)
    assert "threadgroup float As" in src and "threadgroup float Bs" in src
    assert src.count("threadgroup_barrier") == 2
    assert mm.barrier_scopes_emitted() == ["THREADGROUP", "THREADGROUP"]


def test_threadgroup_barriers_are_executable_because_air_emits_them():
    """Barrier REPRESENTATION is refused for elementwise programs, but a matmul
    LOWERS its own barriers, so the scope is executable there. The distinction is
    the point: refusal tracks what the backend can emit, not what AIR can name."""
    mm = air.AirMatmul("m", 64, 64, 64)
    ok, _ = mm.executable_on_metal_backend()
    assert ok
    p = air.AirProgram("b", [air.AirTensor("x", (8,))],
                       [air.AirOp("relu", ("x",), "z")], "z",
                       barriers=[air.AirBarrier("THREADGROUP")])
    assert not p.executable_on_metal_backend()[0]


def test_a_matmul_operand_in_a_foreign_domain_is_refused():
    with pytest.raises(NotImplementedError, match="HUMF"):
        air.lower_matmul_to_msl(
            air.AirMatmul("v", 64, 64, 64, a_domain="NVIDIA_VRAM_DIRECT"))


@pytest.mark.parametrize("tile,msg", [(64, "exceeds 32"), (3, "power of two"),
                                      (0, "power of two")])
def test_impossible_tiles_are_refused(tile, msg):
    with pytest.raises(ValueError, match=msg):
        air.AirMatmul("t", 64, 64, 64, tile=tile).validate()


def test_tile_is_a_specialization_the_forge_can_tune():
    """Different tiles produce different kernels, which is what makes tuning possible
    at the AIR level rather than by editing MSL."""
    srcs = {t: air.lower_matmul_to_msl(air.AirMatmul("m", 64, 64, 64, tile=t))
            for t in (8, 16, 32)}
    assert len(set(srcs.values())) == 3
    assert "As[8][8]" in srcs[8] and "As[32][32]" in srcs[32]


# --- simdgroup matmul strategy ---

def test_strategies_emit_different_kernels_and_launches():
    t = air.AirMatmul("m", 64, 64, 64, tile=16, strategy="tiled")
    s = air.AirMatmul("m", 64, 64, 64, strategy="simdgroup")
    assert air.lower_matmul_to_msl(t) != air.lower_matmul_to_msl(s)
    assert "simdgroup_multiply_accumulate" in air.lower_matmul_to_msl(s)
    assert t.launch() != s.launch()


def test_simdgroup_emits_no_barrier_because_a_simd_runs_in_lockstep():
    assert air.AirMatmul("m", 64, 64, 64, strategy="simdgroup").barrier_scopes_emitted() == []
    assert air.AirMatmul("m", 64, 64, 64, strategy="tiled").barrier_scopes_emitted() == [
        "THREADGROUP", "THREADGROUP"]


@pytest.mark.parametrize("dim", [("m", 60), ("k", 60), ("n", 60)])
def test_simdgroup_refuses_shapes_that_are_not_multiples_of_eight(dim):
    kw = {"m": 64, "k": 64, "n": 64}
    kw[dim[0]] = dim[1]
    with pytest.raises(ValueError, match="multiple of 8"):
        air.AirMatmul("m", strategy="simdgroup", **kw).validate()


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown matmul strategy"):
        air.AirMatmul("m", 64, 64, 64, strategy="wishful").validate()


def test_tiled_has_no_shape_constraint_so_it_is_not_deleted():
    """simdgroup wins on speed but cannot take arbitrary shapes, which is why both
    strategies are kept rather than one replacing the other."""
    air.AirMatmul("m", 60, 60, 60, strategy="tiled").validate()


# --- register blocking ---

def test_block_2_emits_four_accumulators_and_four_macs():
    s = air.lower_matmul_to_msl(air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=2))
    assert s.count("make_filled_simdgroup_matrix") == 4
    assert s.count("simdgroup_multiply_accumulate") == 4
    assert s.count("simdgroup_load") == 4      # 4 loads feeding 4 MACs, not 2 feeding 1
    assert s.count("simdgroup_store") == 4


def test_block_1_is_the_unblocked_kernel():
    s = air.lower_matmul_to_msl(air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=1))
    assert s.count("simdgroup_multiply_accumulate") == 1
    assert s.count("simdgroup_load") == 2


def test_blocking_changes_the_launch_geometry():
    a = air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=1)
    b = air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=2)
    assert a.launch() != b.launch()


def test_block_2_refuses_shapes_it_cannot_cover():
    with pytest.raises(ValueError, match="multiple of 16"):
        air.AirMatmul("m", 24, 64, 64, strategy="simdgroup", block=2).validate()


def test_unimplemented_block_is_refused():
    with pytest.raises(ValueError, match="only 1 and 2"):
        air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=3).validate()


# --- reduction ---

def test_reduce_emits_a_barrier_and_a_simd_reduction():
    s = air.lower_reduce_to_msl(air.AirReduce("r", 1024, "sum"))
    assert "simd_sum" in s and s.count("threadgroup_barrier") == 1
    assert "simd_max" in air.lower_reduce_to_msl(air.AirReduce("r", 1024, "max"))


def test_reduce_refuses_an_unknown_op_and_a_bad_threadgroup():
    with pytest.raises(ValueError, match="unknown reduce op"):
        air.AirReduce("r", 64, "median").validate()
    with pytest.raises(ValueError, match="multiple of the"):
        air.AirReduce("r", 64, "sum", threadgroup=100).validate()


def test_reduce_partial_count_covers_every_element():
    for n in (1, 255, 256, 257, 1 << 20):
        rd = air.AirReduce("r", n, "sum", threadgroup=256)
        assert rd.partials() * 256 >= n


# --- fused softmax ---

def test_softmax_emits_two_barriers_and_both_simd_reductions():
    s = air.lower_softmax_to_msl(air.AirSoftmax("s", 8, 64))
    assert s.count("threadgroup_barrier") == 2
    assert "simd_max" in s and "simd_sum" in s
    assert air.AirSoftmax("s", 8, 64).barrier_scopes_emitted() == ["THREADGROUP"] * 2


def test_softmax_subtracts_the_row_max_for_stability():
    """Without it, exp of a large logit overflows to inf and the row becomes nan."""
    s = air.lower_softmax_to_msl(air.AirSoftmax("s", 8, 64))
    assert "rowmax" in s and "exp(x[base + c] - rowmax)" in s


def test_softmax_refuses_a_bad_threadgroup_and_empty_shapes():
    with pytest.raises(ValueError, match="multiple of 32"):
        air.AirSoftmax("s", 8, 64, threadgroup=48).validate()
    with pytest.raises(ValueError, match="must be positive"):
        air.AirSoftmax("s", 0, 64).validate()


def test_softmax_launches_one_threadgroup_per_row():
    sm = air.AirSoftmax("s", 37, 64, threadgroup=128)
    grid, tg = sm.launch()
    assert grid == (37 * 128, 1, 1) and tg == (128, 1, 1)


# --- fused attention ---

def test_attention_emits_four_barriers_and_holds_scores_in_threadgroup_memory():
    s = air.lower_attention_to_msl(air.AirAttention("a", 64, 64, 32))
    assert s.count("threadgroup_barrier") == 4
    assert "threadgroup float scores[64]" in s
    assert air.AirAttention("a", 64, 64, 32).barrier_scopes_emitted() == ["THREADGROUP"] * 4


def test_attention_refuses_a_sequence_that_will_not_fit_in_threadgroup_memory():
    """This kernel is not flash-attention: it holds the whole score row, so a long
    sequence must be refused rather than silently truncated."""
    with pytest.raises(ValueError, match="online softmax"):
        air.AirAttention("a", 10, 100_000, 64).validate()


def test_causal_flag_changes_the_emitted_kernel():
    plain = air.lower_attention_to_msl(air.AirAttention("a", 64, 64, 32, causal=False))
    causal = air.lower_attention_to_msl(air.AirAttention("a", 64, 64, 32, causal=True))
    assert "-INFINITY;" in causal and plain != causal


def test_attention_reports_the_materialisation_it_avoids():
    at = air.AirAttention("a", 1024, 1024, 64)
    assert at.materialised_bytes_avoided() == 2 * 1024 * 1024 * 4


# --- atomic reduction strategy ---

def test_atomic_strategy_emits_an_atomic_add_and_one_barrier():
    s = air.lower_reduce_to_msl(air.AirReduce("r", 1024, "sum", strategy="atomic"))
    assert "atomic_fetch_add_explicit" in s
    assert s.count("threadgroup_barrier") == 1


def test_atomic_strategy_refuses_max():
    """Metal has no float atomic max here, so max must stay two-stage rather than
    silently producing a wrong answer."""
    with pytest.raises(ValueError, match="sum only"):
        air.AirReduce("r", 64, "max", strategy="atomic").validate()


def test_unknown_reduce_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown reduce strategy"):
        air.AirReduce("r", 64, "sum", strategy="wishful").validate()


def test_two_stage_remains_available_for_max():
    air.AirReduce("r", 64, "max", strategy="two_stage").validate()


# ---------------------------------------------------------------- scan (G046)

def test_scan_refuses_max_on_the_simd_prefix_strategy():
    """Metal has simd_prefix_inclusive_sum and _product but no prefix max. A strategy
    that quietly became a different strategy would make its own benchmark a lie."""
    with pytest.raises(ValueError, match="prefix max"):
        air.AirScan("s", 1024, "max", True, 256, "simd_prefix").validate()
    air.AirScan("s", 1024, "max", True, 256, "hillis_steele").validate()   # legal


def test_scan_refuses_exclusive_max_and_says_it_is_unbuilt_not_impossible():
    with pytest.raises(ValueError, match="NOT IMPOSSIBLE"):
        air.AirScan("s", 1024, "max", False, 256, "hillis_steele").validate()


def test_scan_strategies_emit_different_barrier_counts():
    """simd_prefix does its 32-wide scan inside the SIMD unit and needs 2 barriers;
    hillis_steele holds the block in threadgroup memory and pays 2 per doubling step.
    Conflating them would hide what the vendor primitive actually buys."""
    s = air.AirScan("s", 4096, "sum", True, 256, "simd_prefix")
    h = air.AirScan("s", 4096, "sum", True, 256, "hillis_steele")
    assert len(s.barrier_scopes_emitted()) == 2
    assert len(h.barrier_scopes_emitted()) == 1 + 2 * 8      # log2(256) steps
    assert "simd_prefix_inclusive_sum" in air.lower_scan_block_to_msl(s)
    assert "simd_prefix_inclusive_sum" not in air.lower_scan_block_to_msl(h)


def test_scan_blocks_covers_a_non_multiple():
    assert air.AirScan("s", 4099, "sum", True, 256).blocks() == 17


def test_scan_executes_and_matches_an_f64_oracle():
    pytest.importorskip("mlx.core")
    import numpy as np
    rng = np.random.default_rng(5)
    for n in (7, 1000, 4099, 1 << 16):
        x = rng.standard_normal(n).astype(np.float32)
        ref = np.cumsum(x.astype(np.float64))
        scale = float(np.max(np.abs(ref)))
        for strat in ("simd_prefix", "hillis_steele"):
            g = np.array(air.execute_scan(air.AirScan("t", n, "sum", True, 256, strat), x))
            assert np.max(np.abs(g - ref)) / scale < 1e-5, (n, strat)
        e = np.array(air.execute_scan(air.AirScan("t", n, "sum", False, 256), x))
        assert np.max(np.abs(e - (ref - x))) / scale < 1e-5
        m = np.array(air.execute_scan(air.AirScan("t", n, "max", True, 256, "hillis_steele"), x))
        assert np.array_equal(m, np.maximum.accumulate(x))


def test_a_block_local_scan_alone_is_wrong():
    """The structural fact that makes scan harder than reduction: a block cannot
    finish without every earlier block's total. Phase one ALONE must be wrong, or the
    three-phase shape is unmotivated and the correctness test proves nothing."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    n = 1 << 16
    rng = np.random.default_rng(9)
    x = rng.standard_normal(n).astype(np.float32)
    ref = np.cumsum(x.astype(np.float64))
    sc = air.AirScan("c", n, "sum", True, 256, "simd_prefix")
    k = mx.fast.metal_kernel(name="t_blockonly", input_names=["x"],
                             output_names=["out", "sums"],
                             source=air.lower_scan_block_to_msl(sc),
                             ensure_row_contiguous=True)
    y, _ = k(inputs=[mx.array(x)], grid=(sc.blocks() * 256, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(n,), (sc.blocks(),)],
             output_dtypes=[mx.float32, mx.float32])
    mx.eval(y)
    err = np.max(np.abs(np.array(y) - ref)) / float(np.max(np.abs(ref)))
    assert err > 1e-3, f"block-local scan should be badly wrong, got {err}"


def test_kernel_cache_returns_the_same_object():
    """Building the wrapper per call put MLX's kernel lookup inside timing loops and
    showed up as 20%+ IQR. Pinned because it is a measurement correctness property,
    not a micro-optimization."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    src = air.lower_scan_offset_to_msl(air.AirScan("k", 1024, "sum", True, 256))
    a = air.metal_kernel(mx, name="t_cache", input_names=["y", "off"],
                         output_names=["out"], source=src, ensure_row_contiguous=True)
    b = air.metal_kernel(mx, name="t_cache", input_names=["y", "off"],
                         output_names=["out"], source=src, ensure_row_contiguous=True)
    assert a is b


# ------------------------------------------------------------ graphs (G046/G045)

def _body(n, src):
    return f"uint i=thread_position_in_grid.x; if(i<{n}u) out[i]={src}[i]*1.5f+0.25f;"


def _chain(n, k):
    g = air.AirGraph(f"c{k}", externals=["x"]); prev = "x"
    for j in range(k):
        g.nodes.append(air.AirGraphNode(f"n{j}", _body(n, prev), [prev], n)); prev = f"n{j}"
    return g


def _fan(n, k):
    g = air.AirGraph(f"f{k}", externals=["x"])
    for j in range(k):
        g.nodes.append(air.AirGraphNode(f"n{j}", _body(n, "x"), ["x"], n))
    return g


def test_graph_depth_and_width_separate_ordering_from_independence():
    """The two halves of what CUDA graphs and streams offer. A chain can only be
    batched; a fan is the only shape where concurrency could possibly help."""
    c, f = _chain(4096, 8), _fan(4096, 8)
    assert (c.serial_depth(), c.width()) == (8, 1)
    assert (f.serial_depth(), f.width()) == (1, 8)
    assert c.submissions() == f.submissions() == 1


def test_graph_refuses_a_forward_reference():
    g = air.AirGraph("bad", externals=["x"])
    g.nodes.append(air.AirGraphNode("a", _body(64, "later"), ["later"], 64))
    with pytest.raises(ValueError, match="neither an external nor a node"):
        g.validate()


def test_graph_refuses_duplicate_node_names():
    g = air.AirGraph("dup", externals=["x"])
    g.nodes += [air.AirGraphNode("a", _body(64, "x"), ["x"], 64),
                air.AirGraphNode("a", _body(64, "x"), ["x"], 64)]
    with pytest.raises(ValueError, match="unique"):
        g.validate()


def test_graph_executes_a_chain_in_order():
    """A chain of 1.5x+0.25 is order-sensitive: a dropped or reordered node changes
    the answer, so matching numpy proves the dependency edges were honoured."""
    pytest.importorskip("mlx.core")
    import numpy as np
    x = np.random.default_rng(1).standard_normal(4096).astype(np.float32)
    env = air.execute_graph(_chain(4096, 4), {"x": x})
    ref = x
    for _ in range(4):
        ref = ref * 1.5 + 0.25
    assert np.max(np.abs(np.array(env["n3"]) - ref)) < 1e-5


def test_eager_mode_is_a_control_not_a_feature():
    """eager=True exists only so the batched number has a baseline. It must produce
    the SAME answer -- if it did not, the comparison would be between two different
    computations rather than two submission strategies."""
    pytest.importorskip("mlx.core")
    import numpy as np
    x = np.random.default_rng(4).standard_normal(4096).astype(np.float32)
    a = air.execute_graph(_chain(4096, 4), {"x": x})["n3"]
    b = air.execute_graph(_chain(4096, 4), {"x": x}, eager=True)["n3"]
    assert np.array_equal(np.array(a), np.array(b))
