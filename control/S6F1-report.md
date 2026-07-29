# S6 F1 — Assertion ledger apparatus (paired with F2 r6)

**Base:** `ea33af24ea34a4be43c3ed1ea28b1dd6171b5314`
**Worktree:** `s6-f1-f2-atomic-builder-20260729-164118`
**Commit:** NONE
**Date:** 2026-07-29
**Paired F2 revision:** 6

## Seal (immutable)

| Item | Value |
|------|-------|
| Cases | **4623** |
| sha256 | `5cd38adad4438db156086cacf49e405d0e9fdc050d95113c8e9cf359e3f12011` |
| Sealed commit | `ea33af24ea34a4be43c3ed1ea28b1dd6171b5314` |
| Ledger `--check` | PASS |

## Apparatus

- `tools/verify/case_extract.py` — extraction + seal check
- `tools/verify/test_case_manifest.py` — exclusive ownership gate
- unit tests: **21 OK**
- empty scaffold gate: unaccounted=**4623** (required fail)

## Pairing note (r6)

F1 apparatus LOC remains **+2788** vs base (counted py/md). F2 r6 expands C4
ownership to complete offline GLM52 densify-in-place but does **not** clear the
F2 floor −3400. Combined active LOC **+1627**. See `control/S6F2-report.md` and
`/private/tmp/HAWKING_S6_F1_F2_REVISION6_20260729.md`.

## Retracted

Any claim that F1+F2 combined already nets ≤0 under r5 partial C4 or r6
expanded C4 without further authorized scope is retracted.

## Enabling-branch note

Paired F2 floor remains `F2_FLOOR_STATUS=OPEN_BLOCKED`. This F1 seal is for
off-main enabling use as the parent of C-HIST-R1 / BRT subject-release receipts.
Standalone main integration of F1+F2 alone is **FORBIDDEN**.
