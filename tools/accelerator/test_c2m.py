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
