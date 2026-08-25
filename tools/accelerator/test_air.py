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
