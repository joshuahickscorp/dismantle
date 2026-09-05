# Measured frontier input — a prefix RESTORE costs more than no checkpoint — 2026-09-05

Same process, same length, same machine. Only prefix sharing varies.

## THE MEASUREMENT

Three calls, one resident process. Calls 2 and 3 share their ENTIRE leading text with the
call before them; only the final instruction line differs.

    call 1   cold, no prefix to reuse                3,911 tok    66.46 s    58.84 tok/s
    call 2   shares the full prefix with call 1      3,912 tok   150.62 s    25.97 tok/s
    call 3   shares the full prefix with call 2      3,911 tok     0.80 s  4905.66 tok/s

Control, from the same session, with prompts built from a DISJOINT vocabulary so nothing is
shared:

    long then short, no shared prefix, one process
        call 1 long   14,487 tok   401.17 s
        call 2 short   3,911 tok    60.35 s     <- FASTER than the 66.28 s fresh control
        call 3 short   3,911 tok    60.32 s

So a warm process is not slow. A process that RESTORES A CHECKPOINT is.

## WHAT THIS MEANS

    RESTORE path   2.27x SLOWER than a cold prefill of the same prompt
    APPEND path    ~83x FASTER than cold

The optimization has two paths and they point in opposite directions. The pure-append path
(`reuse == resident_context.len()`, the prompt is a strict extension of what the session
already holds) is enormous and real. The checkpoint-restore path, taken when the prompt
DIVERGES from the session but still matches the stored checkpoint, is a PESSIMIZATION on
this body.

This is the mechanism behind the 4.5x "degradation" measured across the 9-cell retrieval
run. Those cells were the same filler with a needle moved, so consecutive prompts diverged
at the needle and repeatedly took the restore path.

## WHY IT IS NOT MERELY ROUTE SELECTION

A restore sets `reuse != 0`, which `qwen38_batched_prefill_allowed` uses to refuse batched
prefill. But the COLD call also takes a snapshot, so `snapshot_at.is_some()` and it is
sequential too. Both calls above ran the sequential route. The 2.27x is ON TOP of route
selection: the restore itself is expensive.

## THE QUESTIONS FOR THIS MISSION

1. WHY is a restore more expensive than a cold prefill of the same tokens? A restore should
   SKIP work. Find what it does instead. `restore_prefix` and the code path that follows it
   in crates/hawking-core/examples/ascension_qwen38_resident.rs and the session's
   restore implementation are the place to look.

2. Given the answer, is the right change to make restore cheap, or to REFUSE a restore whose
   measured cost exceeds a cold prefill? A checkpoint that is slower than not having one
   should not be used, and the decision is measurable per call: restored prefix length
   versus total prompt length.

3. Does the pure-append path's ~83x survive at HCLI's real prompt sizes and shapes? If it
   does, the highest-value change may be making HCLI's packets APPEND-SHAPED -- stable
   prefix, new material only at the end -- rather than tuning the restore.

## CONSTRAINTS

- Report fresh and reused prefill SEPARATELY. The 0.80 s call is REUSED, not fresh, and
  quoting 4,905 tok/s as a prefill rate would be a lie.
- Token identity is non-negotiable; a restore that changes the answer is a failure.
- Every file in receipts/sovereign/VERIFIER_MANIFEST.json is PROTECTED.
- Smallest executable change. Name a test that fails before and passes after.
