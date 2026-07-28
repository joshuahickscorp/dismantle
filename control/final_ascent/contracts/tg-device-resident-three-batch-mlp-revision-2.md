# TG device-resident three-batch MLP — Revision 2 live-Metal correction

Revise the Revision-1 candidate in place. Preserve the controlling base
`ba2ca65b1765e833ec381b454ee1d68b48534656`, authorized seven implementation
files plus one focused test, default-off/non-wave/no-ICB constraints,
abortable ownership, exact arithmetic, poison/reset, physical evidence, and
false fences.

The only authorized Revision-1 predecessor blobs are:

- `gravity_glm.rs`:
  `0f07029031ef3ca94ab967b7bff69f912d5d599a`;
- `gravity_glm_resident.rs`:
  `ca7ff0cb15c913e8bf01a891775bdc5ef84ff997`;
- `metal/mod.rs`:
  `ee137f4824e8b090684b6f57c0a3f74b0889ab8b`;
- `cost_ledger.rs`:
  `88ae09e370667cb20722c99338fdebc93544d500`;
- `numeric_parity.rs`:
  `db4dbf2222999b3a7d25a68f76e83e664c6801c4`;
- `gravity_pq.metal`:
  `b555803b9c0246739bc493815f721ae5bb273776`;
- focused test:
  `66ef23f6eaebb883875232aa3cc08d718a7c1ede`.

Refuse before editing if any identity differs. Remove `.serena` additions.

## 1. Frozen live failure

An unsandboxed local Metal run on Apple M3 Ultra executed the candidate and
failed:

```text
live_tiny_three_batch_parity_topology_and_probes
prompt 0
device meaningful_rel = 6.194e-3
V2.1 bound = 1.000e-5
```

The same worktree with
`HAWKING_GLM_GPU_DEVICE_RESIDENT_THREE_BATCH_MLP` unset passes
`resident_matches_host_state_over_several_prompts` with bit-identical resident
logits. Therefore this is a candidate MLP arithmetic/order/lifetime defect,
not an unavailable Metal device or pre-existing resident-path difference.

Revision 1 is rejected. Do not weaken Numeric Parity V2.1, change the oracle,
label the failure diagnostic, or count the topology as an acceleration.

## 2. Localize the first divergence

Add a source-body-free, diagnostic-only fixture mode that compares the same
resident baseline with the flag off and candidate with the flag on. Both modes
must use identical resident attention/router/head flags, tensors, prompts,
device, and command mode. The governing comparison is not host-state versus
resident.

At the real ordinary MLP insertion point, capture and compare, for each layer
and expert in exact execution order:

- normalized MLP input;
- gate and up outputs;
- SiLU product;
- down output;
- each prefix of the ordered weighted accumulation;
- shared-last addition;
- residual before and after.

This diagnostic may perform readback only in an explicit ledger-on
non-promotable run. Bind layer, expert, tensor hashes/address generations,
route IDs/weights/order, command/fence generations, and exact f32 bits.
Mechanically report the first differing stage and element. No shipping
candidate path may add these readbacks.

Test both native BF16 and direct-u8 PQ fixtures. If a codec cannot establish the
required exact authority, refuse that codec before mutation rather than
guessing.

## 3. Correctness requirements

Fix the load-bearing cause. Preserve:

- gate/up/down matvec shapes, transposition, codec decode, and accumulation
  order exactly matching the resident baseline;
- dependency ordering between gate/up, activation, down, ordered accumulation,
  shared-last addition, and residual;
- no destination alias, no scratch reuse before its completion fence, and no
  address-generation drift;
- exact route IDs, weights, execution order, and shared scale;
- the baseline's actual SiLU, multiply/add/FMA contraction, signed-zero,
  subnormal/FTZ, narrowing, NaN/infinity, and residual policies.

Metal comments or `volatile` are not arithmetic evidence. Execute bit-vector
and complete insertion-point comparisons on the real device. If the existing
Metal compiler cannot guarantee the baseline policy, use an explicit kernel or
refuse promotion.

The governing matrix must include:

- same-resident baseline/candidate over all tiny prompts;
- per-stage first-divergence fixture;
- complete logits V2.1 and exact greedy/top-k/router/DSA decisions;
- explicit `three_batch_hit > 0`, `three_batch_fallback == 0`, and all
  wave/table/replay/ICB probes zero;
- native BF16, direct-u8 PQ, unsupported codec, activation-aware refusal;
- alias, undersize, stale generation, injected failure at every step,
  abort-before-commit, failure-after-submit poison, and verified reset;
- no partial residual/KV/sequence/trace publication on failure.

Do not use a widened f32 vector as original-input FP64 authority. Where true
FP64 authority is unavailable, require bit identity to the frozen same-device
resident baseline and label the FP64/TG qualification unavailable.

## 4. Honest live topology and latency

Only after parity passes, run the frozen protocol:

- `HAWKING_TCB_TRACE=off`;
- 20 warmups and 200 measured iterations per mode;
- randomized/interleaved paired order over the same prompts;
- timing ledger off, separate topology ledger on;
- nearest-rank p50/p95, physical command buffers, waits, dispatches,
  allocations, D2H/H2D bytes, and GPU completion status.

The current test builds separate per-mode blocks despite constructing an
interleaving vector. Correct it so execution is actually interleaved and paired.

Accept only if both p50 and p95 do not regress outside the frozen 2% paired
tolerance and the measured physical topology matches the claimed two
dependency boundaries. A wait reduction with worse wall time is a negative
receipt, as with expert-wave. Fixture results are never `BASE_TRUE_TPS` or a
TG milestone.

## 5. Exit

Return exact Git blobs and SHA-256 identities, complete local test output, the
first-divergence diagnosis and correction, and the full live Metal parity,
topology, and timing table. Keep every product flag default off and all
authorization/MOP/HIDE/TG/capable-provider claims false.
