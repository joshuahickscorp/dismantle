"""C2M pins. FRONT C (G045). The refusals matter more than the successes: a silent
mistranslation is worse than a rejection."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import c2m  # noqa: E402

IDX = "int i = blockIdx.x * blockDim.x + threadIdx.x;"


def k(body, params="const float* a, const float* b, float* c, int n"):
    return f"__global__ void t({params}) {{ {IDX} {body} }}"


def test_translates_the_four_supported_forms():
    for expr, op in [("a[i] + b[i]", "add"), ("a[i] * b[i]", "mul"),
                     ("a[i] - b[i]", "sub"), ("fmaxf(a[i], 0.0f)", "relu")]:
        t = c2m.translate(k(f"if (i < n) c[i] = {expr};"), elements=64)
        assert t.program.ops[0].kind == op
        assert t.tier == "C2M-T0"


@pytest.mark.parametrize("body,token", [
    ("__shared__ float s[4]; c[i] = a[i] + b[i];", "shared memory"),
    ("for (int j=0;j<4;j++) c[i] = a[i] + b[i];", "loops"),
    ("while (n) c[i] = a[i] + b[i];", "loops"),
    ("atomicAdd(&c[i], b[i]);", "atomics"),
    ("__syncthreads(); c[i] = a[i] + b[i];", "block synchronization"),
    ("printf(\"x\"); c[i] = a[i] + b[i];", "device printf"),
])
def test_unsupported_constructs_are_refused_by_name(body, token):
    with pytest.raises(c2m.C2MRefusal) as e:
        c2m.translate(k(body), elements=64)
    assert token in str(e.value)


def test_fp64_is_refused():
    with pytest.raises(c2m.C2MRefusal, match="fp64"):
        c2m.translate(k("c[i] = a[i] + b[i];",
                        params="const double* a, const double* b, double* c, int n"),
                      elements=64)


def test_an_unrecognised_expression_is_refused_not_guessed():
    with pytest.raises(c2m.C2MRefusal, match="outside the C2M-T0 pattern set"):
        c2m.translate(k("c[i] = sinf(a[i]);"), elements=64)


def test_missing_thread_index_is_refused():
    src = "__global__ void t(const float* a, float* c, int n) { c[i] = a[i] + a[i]; }"
    with pytest.raises(c2m.C2MRefusal, match="thread index"):
        c2m.translate(src, elements=64)


def test_multiple_stores_are_refused():
    with pytest.raises(c2m.C2MRefusal, match="more than one store"):
        c2m.translate(k("c[i] = a[i] + b[i]; c[i] = a[i] * b[i];"), elements=64)


def test_a_non_pointer_operand_is_refused():
    src = ("__global__ void t(const float* a, float n, float* c) { " + IDX +
           " c[i] = a[i] + n[i]; }")
    with pytest.raises(c2m.C2MRefusal, match="not a pointer parameter"):
        c2m.translate(src, elements=64)


def test_conformance_never_claims_a_tier_it_did_not_earn():
    empty = c2m.conformance([])
    assert empty["tier_claimed"] == "NONE"
    got = c2m.conformance([{"matches_oracle": True}])
    assert got["tier_claimed"] == "C2M-T0"
    for tier in ("C2M-T1", "C2M-T2", "C2M-T3", "C2M-T4", "C2M-T5"):
        assert got["higher_tiers"][tier].startswith("NOT CLAIMED")


def test_conformance_states_it_is_not_a_cuda_differential():
    c = c2m.conformance([{"matches_oracle": True}])
    assert c["is_a_cuda_differential"] is False
    assert c["oracle"] == "numpy on CPU"


# ------------------------------------------------------------- T1 runtime (G045)

import cuda_runtime as cr  # noqa: E402

_KERNELS = {"vadd": "__global__ void vadd(const float* a, const float* b, float* c, int n){"
                    " int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }"}
_SAFE = """
cudaMalloc((void**)&d_a, N*sizeof(float));
cudaMalloc((void**)&d_b, N*sizeof(float));
cudaMalloc((void**)&d_c, N*sizeof(float));
cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
vadd<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
cudaDeviceSynchronize();
cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
"""
_HAZARD = _SAFE.replace("vadd<<<", "h_a[0] = 999.0f;\nvadd<<<")


@pytest.mark.parametrize("bad,frag", [
    ("cudaStreamCreate(&s);", "streams"),
    ("cudaMemcpyAsync(a,b,4,cudaMemcpyHostToDevice,0);", "asynchronous"),
    ("cudaMallocManaged((void**)&p, 4);", "managed memory"),
    ("cudaMemset(d_a, 0, 4);", "memset"),
    ("int x = foo();", "outside the C2M-T1 subset"),
])
def test_t1_refuses_by_name(bad, frag):
    with pytest.raises(c2m.C2MRefusal, match=frag):
        cr.parse_host(bad)


def test_may_delete_copies_names_the_offending_statement():
    """The condition on S015 §9. A host write after the copy makes the alias
    observationally different from the snapshot, so the deletion is refused."""
    assert cr.parse_host(_SAFE).may_delete_copies()[0] is True
    ok, why = cr.parse_host(_HAZARD).may_delete_copies()
    assert ok is False and "h_a" in why and "999.0f" in why


def test_a_write_after_the_launch_does_not_block_the_deletion():
    """The guard must be about the SNAPSHOT WINDOW, not about host writes in
    general -- otherwise it would refuse programs that are perfectly safe."""
    after = _SAFE.replace("cudaDeviceSynchronize();", "cudaDeviceSynchronize();\nh_a[0] = 5.0f;")
    assert cr.parse_host(after).may_delete_copies()[0] is True


def test_both_modes_agree_on_a_safe_program():
    pytest.importorskip("mlx.core")
    import numpy as np
    n = 4096
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal(n).astype(np.float32) for _ in range(2))
    arr = lambda: {"h_a": a.copy(), "h_b": b.copy(), "h_c": np.zeros(n, np.float32)}
    f = cr.execute_host(_SAFE, _KERNELS, arr(), elements=n, mode="FAITHFUL")
    u = cr.execute_host(_SAFE, _KERNELS, arr(), elements=n, mode="UNIFIED")
    assert np.array_equal(f["host"]["h_c"], u["host"]["h_c"])
    assert np.max(np.abs(f["host"]["h_c"] - (a + b))) < 1e-5
    assert (f["copies_performed"], u["copies_performed"]) == (3, 0)


def test_the_hazard_is_real_and_the_guard_is_necessary():
    """A refusal nobody has watched be NECESSARY is indistinguishable from one that
    is merely cautious. Bypassing the guard must produce a WRONG answer."""
    pytest.importorskip("mlx.core")
    import numpy as np
    n = 4096
    rng = np.random.default_rng(3)
    a, b = (rng.standard_normal(n).astype(np.float32) for _ in range(2))
    arr = lambda: {"h_a": a.copy(), "h_b": b.copy(), "h_c": np.zeros(n, np.float32)}
    f = cr.execute_host(_HAZARD, _KERNELS, arr(), elements=n, mode="FAITHFUL")
    u = cr.execute_host(_HAZARD, _KERNELS, arr(), elements=n, mode="UNIFIED", _unsafe_ok=True)
    assert abs(f["host"]["h_c"][0] - u["host"]["h_c"][0]) > 100
    assert np.array_equal(f["host"]["h_c"][1:], u["host"]["h_c"][1:])  # one element only
    with pytest.raises(c2m.C2MRefusal):
        cr.execute_host(_HAZARD, _KERNELS, arr(), elements=n, mode="UNIFIED")


def test_uninitialised_device_read_is_refused():
    pytest.importorskip("mlx.core")
    import numpy as np
    src = ("cudaMalloc((void**)&d_a, N*sizeof(float));\n"
           "cudaMalloc((void**)&d_b, N*sizeof(float));\n"
           "cudaMalloc((void**)&d_c, N*sizeof(float));\n"
           "vadd<<<1,256>>>(d_a, d_b, d_c, N);\n")
    with pytest.raises(c2m.C2MRefusal, match="never written"):
        cr.execute_host(src, _KERNELS, {"h_a": np.zeros(4, np.float32)},
                        elements=4, mode="FAITHFUL")


def test_t1_is_claimed_only_when_both_modes_agree():
    assert cr.conformance_t1([])["tier_claimed"] == "C2M-T0"
    assert cr.conformance_t1([{"matches_oracle": True, "both_modes_agree": False}]
                             )["tier_claimed"] == "C2M-T0"
    c = cr.conformance_t1([{"matches_oracle": True, "both_modes_agree": True}])
    assert c["tier_claimed"] == "C2M-T1"
    assert "FRONTEND" in c["higher_tiers"]["C2M-T2"] or "C2M frontend" in c["higher_tiers"]["C2M-T2"]
