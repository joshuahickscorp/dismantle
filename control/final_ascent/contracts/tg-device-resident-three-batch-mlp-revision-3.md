# TG device-resident three-batch MLP — Revision 3 device-only correction

Revise the rejected Revision-2 candidate in place. Preserve the controlling
base, authorized surfaces, default-off/non-wave/no-table/no-replay/no-ICB
isolation, abortable ownership, poison/reset, physical evidence, source-body-
free fixtures, and every false authorization/MOP/HIDE/TG fence.

The only authorized Revision-2 predecessor blobs are:

- `crates/hawking-core/src/gravity_glm.rs`:
  `c28577dc9f1e4f3db104ca1eeab0ba62f49eefe3`;
- `crates/hawking-core/src/gravity_glm_resident.rs`:
  `f2369f9ed576639d6268fd9b03e55764b1c6a8bc`;
- `crates/hawking-core/src/metal/mod.rs`:
  `ee137f4824e8b090684b6f57c0a3f74b0889ab8b`;
- `crates/hawking-core/src/cost_ledger.rs`:
  `88ae09e370667cb20722c99338fdebc93544d500`;
- `crates/hawking-core/src/numeric_parity.rs`:
  `db4dbf2222999b3a7d25a68f76e83e664c6801c4`;
- `crates/hawking-core/shaders/gravity_pq.metal`:
  `b555803b9c0246739bc493815f721ae5bb273776`;
- `crates/hawking-core/tests/gravity_glm_device_resident_three_batch_mlp.rs`:
  `a9c5b96f6dac34c16f293d65d3baa03eb4fa4d56`.

Refuse before editing if any identity differs. Remove `.serena` additions.

## 1. Freeze the Revision-2 negative receipt

The governing unsandboxed Apple M3 Ultra run passed eight tests only after the
candidate inserted a host activation boundary:

```text
topology = two_dependency_boundaries_gate_up__host_silu__down_combine_residual
baseline p50/p95 = 18299/20442 us
candidate p50/p95 = 18174/21352 us
physical commands/token = 36/36
candidate p95 regression = 4.45%
```

Revision 2 is negative and non-promotable. It reads every device-produced gate
and up buffer with `read_f32`, computes `silu_mul_f32_host` in host `Vec`s, and
writes every activation back with `write_f32`. Unified memory does not make
these host-visible mapped reads/writes or the intervening completion wait
device-resident. Do not weaken the 2% live gate, hide the mapped traffic, call
the result a topology win, or derive a TG/TPS claim from the tiny fixture.

## 2. True device-only activation

On the claimed hit path:

- encode the existing non-table `gravity_silu_mul_f32` primitive after each
  gate/up producer and before its down consumer;
- keep gate, up, activation, down, ordered route accumulation, shared-last
  addition, and residual entirely in device buffers;
- remove all hit-path `read_f32(gate)`, `read_f32(up)`,
  `silu_mul_f32_host`, host activation `Vec`, and `write_f32(act)`;
- use one command buffer for the whole MLP when abort/poison ownership can be
  proven, otherwise at most two real device dependency boundaries;
- never enter or alias expert-wave, expert-table, replay, ICB, final-head
  replay, or their scratch/flags/counters;
- preserve native-BF16 and direct-u8-PQ projection layout, transposition,
  tensor/address-generation leases, route order/weights, non-contracted
  weighted accumulation, scale-one shared add, residual order, and exact
  failure ownership.

Prefer the existing shader and encoder. A new kernel is allowed only if an
executed arithmetic counterexample proves the existing primitive insufficient;
return that counterexample and freeze the new kernel identity.

## 3. Honest arithmetic authority

Metal `exp` and host libm `expf` need not be bit-identical. It is therefore
forbidden to claim both a device-only activation and bit identity to the
unchanged host-SiLU flag-off path.

Create an explicit, default-off, independently encoded ordinary-resident
device-SiLU reference mode for the governing candidate comparison:

- the reference must use the ordinary non-three-batch projection/control flow,
  not call the candidate wrapper or share its topology assertions;
- reference and candidate may share only the frozen low-level SiLU primitive;
- compare full residuals/logits bit-for-bit and all greedy, top-k, router, and
  DSA decisions exactly between those two device-SiLU paths;
- separately run Numeric Parity V2.1 against true FP64 original-input
  authority when available, or label FP64/TG qualification unavailable;
- keep the default flag-off resident/host-SiLU path byte-exact and run it as a
  non-regression control;
- report the expected host-SiLU/device-SiLU first differing element as a
  diagnostic, never reseal it into a fake bit-identity gate.

Freeze no-fast-math compilation, explicit divide then multiply, FTZ/subnormal,
signed-zero, NaN/infinity, narrowing, and FMA/contraction policies. Execute
boundary bit vectors on the real Metal device. Refuse before residual mutation
on nonfinite or unsupported arithmetic.

## 4. Physical proof

Instrument mapped shared-buffer reads and writes as physical events, including
the previously uncounted `read_f32` path. A ledger-on live hit must prove:

- zero mapped CPU reads of gate/up/activation/down intermediates;
- zero mapped CPU writes of gate/up/activation/down intermediates;
- zero transfer D2H/H2D bytes for those intermediates;
- zero host activation allocations/computation;
- no completion wait whose sole purpose is host SiLU;
- no per-token weight upload, scratch allocation, command-buffer allocation,
  or bind/rebuild after warmup;
- candidate MLP command buffers/waits strictly fewer than the ordinary
  device-SiLU reference and total commands/token no greater;
- hit count positive, fallback zero, and every wave/table/replay/ICB probe zero.

Counters must come from the physical trace and checked Metal completion status,
not formulas, labels, source inspection, or local helper counters.

## 5. Failure and lifetime matrix

Use real source-body-free Metal fixtures for native BF16 and direct-u8 PQ.
Refuse unsupported and activation-aware codecs before mutation. Exercise stale
source/destination generations, alias/undersize/misalignment, scratch reuse
before fence retirement, and injected failure before and after every encode,
submit, completion, accumulation, shared add, and residual boundary.

Pre-submit failures discard all work. Submitted failures poison every touched
scratch/address generation and publish no residual, KV, sequence, trace, or
receipt. Verified reset/wipe must be required before reuse while neighboring
slots/generations remain healthy.

## 6. Live speed gate

After all correctness and physical gates pass, run the frozen protocol:

- `HAWKING_TCB_TRACE=off`;
- 20 warmups and at least 200 measured iterations per mode;
- truly randomized/interleaved paired execution over identical prompts,
  tensors, routes, device, and flags;
- timing ledger off and a separate topology ledger-on run;
- raw samples plus nearest-rank p50/p95 and paired confidence intervals.

Promotion requires:

- candidate p50 and p95 each no worse than the device-SiLU reference outside
  the frozen 2% tolerance;
- at least one of p50 or p95 has a strictly lower paired 95% confidence bound;
- strict MLP command/wait reduction and no total command increase;
- all parity, decision, isolation, failure, and physical gates green.

Otherwise return a negative/default-off receipt. A topology-only or
fixture-only result is not `BASE_TRUE_TPS`, TG2, TG1, or a production claim.

## 7. Exit

Return exact Git blobs and SHA-256 identities, the first-divergence diagnosis,
full test output, raw live samples, parity/decision matrix, physical counter
table, and positive or negative disposition. Keep real weights, source-body
access, MOP, `HIDE_KERNEL_TURN`, TG/HIDE promotion, capable-provider status,
and all authorization transitions false/default-off.
