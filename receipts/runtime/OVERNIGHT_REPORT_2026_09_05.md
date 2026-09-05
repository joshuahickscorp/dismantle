# HCLI overnight supergoal — report — 2026-09-05

Worktree: .worktrees/ascension   branch: ascension-isolated   HEAD: 3305b0a2e3e7af9a28aa88123b52312290ba136e
Working tree: 0 uncommitted path(s) at the time of writing.

## 1. ACCEPTED MUTATIONS

HCLI-authored, accepted by a verifier that could actually check them: **ZERO**.

One HCLI mutation was accepted at 07:2x and then REVERTED by the supervisor. It deleted the
correctness guard in `qwen38_batched_prefill_allowed` and was accepted on the evidence of
an unrelated Python test, because the verifier had no checker for Rust (section 9). It does
not count and is not counted.

Supervisor-authored substrate repairs, each with negative controls, all committed:

    6a1270bd3  close the FrontierEngine repair lineage (74 -> 64 failures, none new)
    fa01ed0eb  G012 decode producer
    af01fa42d  G012: fresh process per arm (a median over a degrading resident is not a rate)
    47c1d9359  the verifier now compiles Rust; uncheckable SOURCE fails closed
    5ab1065a2  G012 receipt: complete decode 36.67 tok/s
    3305b0a2e  the 9-cell retrieval matrix
    81fae6a91  frontier input: prefill vs prefix checkpoint
    2b2e7b062  frontier input: the KV cache is f32 while the model is bf16
    d5b18b5d7  frontier input: the resident degrades 4.5x across requests

## 2. REJECTED MUTATIONS

    HCLI, guard deletion in qwen38_batched_prefill_allowed   REVERTED by supervisor
    HCLI, non-compiling Rust (mission 3e0ee71e)              REFUSED by cargo_check exit 101
    HCLI, unit naming the protected G005 gate as proof        REFUSED, gate exit 1 (correct:
                                                              the receipt it demands is absent)

## 3. PHYSICAL MEASUREMENTS

Complete decode, marginal method, fresh process, two-point (wall(576)-wall(64))/512:

    36.668 tok/s     repeat 36.635 tok/s     run-to-run spread 0.090%
    aggregate across 2 streams 36.724 tok/s  ->  scaling 1.002x
    swapouts delta during measurement 0

Fresh prefill, fresh process, sequential route (the only route HCLI ever gets):

     7,737 tok    159.9 s    48.4 tok/s
    15,465 tok    444.3 s    34.8 tok/s
    30,921 tok  1,437.0 s    21.5 tok/s
    scaling ~n^1.48 between 8K and 16K, steepening to ~n^1.59 by 32K

Resident degradation across requests in ONE process:

    16K @25%  first call, fresh          446.0 s
    16K @25%  last call after 7 others   445.6 s     (fresh process)
    16K @50%  third call, shared        1,997.8 s     4.5x
    16K @90%  fourth call, shared         >870 s and still running when stopped

KV cache, exact from geometry constants (16 GQA layers x 4 kv_heads x 256 head_dim x 2):

    f32  (today)              131,072 B/token = 128 KB/token
    bf16 (model's own dtype)   65,536 B/token =  64 KB/token

## 4. CAPABILITY CHANGES

Nothing was made faster tonight. What changed is what can be VERIFIED:

  - The mutation loop can now verify Rust. Before 07:33 it could not compile the language
    every remaining Hawking target lives in, so no Rust mutation it ever accepted was
    actually checked.
  - One sovereign gate discharged: G012_resident_perf. 4 of 14 -> 5 of 14.
  - The 9-cell retrieval matrix replaces a stale, wrong capability claim with a measured one.

## 5. CONTEXT / STATE CHANGES

    9 of 9 retrieval cells RETRIEVED, byte-exact, at 8K / 16K / 32K x 25% / 50% / 90%.

The superseded conclusion ("retrieval fails at 16K at all depths") is OVERTURNED. The real
boundary is above 30,921 tokens and was not located. What binds today is the committed
envelope: 9,728 tokens, which RAISES rather than truncates above that limit.

No state reduction was landed. The KV lever is identified exactly (128 -> 64 KB/token) and
belongs to HCLI to author; it is also the enabling condition for G006's 262,144 rung
(34.4 GB of KV at f32 against ~49 GB free plus an ~11 GB body; 17.2 GB at bf16).

## 6. PREFILL / DECODE CHANGES

No improvement landed. The prefill gap is now EXPLAINED rather than open, and it is two
stacked defects, not one:

    F009  a prefix checkpoint (restored OR taken) disqualifies batched prefill, and HCLI
          checkpoints on every call, so HCLI never gets the batched route
    F017  the resident then degrades ~4.5x across the requests of a long mission

This closes the standing "roughly 400 s unaccounted INSIDE the call" question in
FRONTIER_INPUT_2026_09_05.md: that document costed HCLI's calls at 74.8 tok/s, a rate
measured on the batched route HCLI cannot reach.

## 7. TOTAL WORKUNIT WALL IMPROVEMENT

None measured, and the historical baseline is now known to be unsound: every per-WorkUnit
wall taken late in a long mission is inflated by an unknown, position-dependent amount
(F017). The 21,527 s over 41 units and the 525 s/unit average are degraded-process figures.

## 8. LAWS

  L1  A mutation may not be accepted on evidence that cannot cover it. Evidence in one
      language does not verify a change in another. Silence from a missing checker is not
      a passing check.
  L2  Any resident measurement must control for PROCESS HISTORY. One fresh process per arm.
      A median over a degrading series is not a rate.
  L3  A rate divided out of a whole call wall includes model load and prefill. Measure
      decode MARGINALLY between two token counts so both cancel.
  L4  Prefix reuse and batched prefill are mutually exclusive under the current admission
      rule. Any prefill claim must say which route it was measured on.

## 9. SCARS

  S1  A "capability ceiling" was actually a configuration ceiling. The 8K/16K retrieval
      limit came from the wrong renderer plus a window that refused the prompt. Before
      calling a limit physical, check what refused the request.
  S2  Reported field names lie. `prompt_token_count_source: estimated_without_transformers`
      was a MISLABEL over an exact resident count -- hawking_native.py:841 overwrites the
      label whenever the transformers tokenizer is absent, even when an exact count was
      already obtained. Arithmetic caught it: 13108 chars / 3231 tokens = 4.06, not chars/3.
  S3  I made the same class of error twice in one instrument in twenty minutes: dividing a
      whole call wall by its tokens (reported 10.4 tok/s of setup as "decode"), then
      multiplying an already-aggregate rate by the stream count (reported 73.1 tok/s and
      would have published "2 streams nearly double throughput", the exact opposite of the
      truth). Both were caught only by reading the raw arm data. Ask what the number
      measures, then read the rows.

## 10. BLOCKERS AND REOPEN CONDITIONS

  B1  Odyssey gate: fresh prefill >= 100 tok/s. MEASURED BELOW FLOOR at every length
      (48.4 / 34.8 / 21.5). REOPEN when a checkpointed call can use the batched route.
  B2  Odyssey gate: complete decode >= 40 tok/s. MEASURED 36.67, 8.3% short. REOPEN on any
      decode change; note DeltaNet is 33.8% of the token and KV only 11.7%, so decode work
      belongs in DeltaNet, not KV.
  B3  Bootstrap streak (3 consecutive accepted HCLI mutations). NOT CLOSED. Zero accepted
      by the repaired verifier. REOPEN condition: it is now running against a verifier
      worth passing; mission 3e0ee71e is live and detached.
  B4  G006_long_context needs rungs 131,072 and 262,144, not 32K. Blocked on the KV dtype
      change. NOT attempted tonight.
  B5  F017's MECHANISM is unknown. Three candidates recorded (restore-then-sequential,
      high-water sizing, Metal allocation) with a two-call discriminator specified.

## 11. CURRENT RESIDENT

    sealed-3.14, qwen3.8-27b, hawking-native, physical_ebpw 3.1393
    envelope ascension_envelope.hawking.json, max_seq_len 9,728
    binary workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident
    built 2026-09-05 00:20:21, AFTER the last source edit -- it includes uncommitted Rust
    changes and is NOT HEAD.

## 12. CLEAN / DIRTY

    0 uncommitted path(s). Every finding above is committed.

## 13. EXACT HEAD

    3305b0a2e3e7af9a28aa88123b52312290ba136e  on ascension-isolated

## 14. NEXT HIGHEST-VALUE FRONTIER

  1. Let mission 3e0ee71e run. It is detached (ppid 1) and survives this session. It is the
     bootstrap streak's first honest attempt.
  2. F009: batched prefill for the eligible part of a checkpointed prompt. This is the
     critical path to the Odyssey prefill floor, and no other lever reaches 100 tok/s.
  3. F017's mechanism, via the two-call discriminator. It contaminates every wall number
     the campaign has, so it compounds.
  4. F012/F014: KV f32 -> bf16. Exact, large, and the enabling condition for the 262K rung.
     Must be qualified by semantic execution, never reconstruction arithmetic.
  5. NOT next: decode micro-optimization. One stream already saturates the GPU (1.002x on
     two), so there is no scheduling headroom to harvest; the work is inside DeltaNet.
