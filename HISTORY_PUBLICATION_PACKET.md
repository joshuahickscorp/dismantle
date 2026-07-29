# History publication decision packet

Prepared 2026-07-29. **Nothing has been force-pushed. This packet exists so the decision can
be made with the consequences in front of you, not discovered afterward.**

## What was rewritten, and what was not

The local history was rewritten on 2026-07-28. `origin`
(`git@github.com:joshuahickscorp/hawking.git`) still carries the pre-rewrite history, and the
local remote-tracking refs were dropped, so the two have diverged.

Two things changed and nothing else:

**Blobs removed from all history** — ten data artifacts totalling 64 MB, none present in the
working tree at the time:

    HAWKING_BYTE_CATEGORY_LEDGER.json      HAWKING_TG_COST_LEDGER.json
    HAWKING_TG_LEDGER_POSTMEMO.json        HAWKING_LEDGER_BF16.json
    GLM52_AA_PACK_DRY_RUN.json             GLM52_RARE_ROUTE_PILOT_PARTIAL_000{35,86,264}.json
    reports/condense/deepseek_v4_flash/pre_moe_hidden_L05.npy
    colab/data/eagle5_corpus/shard_00039.parquet

**Commit messages** — `grok/` and `codex/` replaced with `lane/` where they appeared in merge
subjects, matching the branch renames done at the same time.

**No author or committer field was touched.** An audit before and after found zero commits
with a non-human author, zero `Co-Authored-By` trailers and zero generated-by footers across
all 1,775 commits then in the tree. There was no AI attribution in this repository to remove.

Five `refs/codex/turn-diffs/checkpoints/...` refs were deleted — an external tool's per-turn
checkpoint namespace that had accumulated inside the repository. Two commits reachable only
from those refs went with them.

`--prune-empty=never` kept every commit record, including the handful whose only content was a
removed blob. `.git` went from 137 MB to 82 MB. The working tree was byte-identical before and
after: `HEAD^{tree}` was `776e640c94f77fab9f9809734252f83dab946b7a` on both sides.

## Impact if published

| | |
|---|---|
| Local branches affected | 149 |
| Tags affected | 63 |
| Commits in rewritten history | 2,558 |
| Every commit hash before 2026-07-28 | changes |
| Existing clones | must re-clone; `git pull` will not reconcile |
| Open pull requests | rebase or recreate against the new history |
| Sealed receipts pinning commit hashes | resolve via the translation map, below |

## Receipt-hash translation coverage

`docs/COMMIT_MAP_20260728.txt` maps **1,819** pre-rewrite commits to their post-rewrite
hashes, old on the left, new on the right.

    grep -i ^<old-hash> docs/COMMIT_MAP_20260728.txt

Known citations of dissolved hashes in tracked files:

| hash | files citing it |
|---|---|
| `fccb6b30` | 11 (HIDE archaeology and campaign status) |
| `0adcab57` | 2 (packs retirement record) |

These receipts were deliberately **not** edited. They are sealed evidence, and rewriting them
to match new hashes would defeat the point of sealing them. The map is the resolution path.

## Clone migration instructions

For anyone holding a clone, after publication:

    # keep nothing local you have not pushed; then
    cd ..
    mv hawking hawking-old
    git clone git@github.com:joshuahickscorp/hawking.git
    # recover any unpushed work from hawking-old by cherry-pick or format-patch

`git pull --rebase` will not work: the histories share no common ancestor after the rewrite.

## Rollback to the old remote history

The pre-rewrite history is preserved in full:

    ~/Downloads/hawking-backup/hawking-pre-rewrite-7632915f.bundle
    ~/Downloads/hawking-backup/hawking-pre-rewrite-08bf0a61.bundle

    git clone hawking-pre-rewrite-08bf0a61.bundle recovered

Since `origin` has not been touched, rollback today requires no action at all — the remote
*is* the old history. After a force-push, rollback means force-pushing the bundle's refs back,
and every clone taken in between would need the same migration a second time.

## Recommendation

**Do not publish yet, and publish only if sub-50 MB clones are worth the disruption.**

The rewrite bought 55 MB of `.git`. Against that: 149 branches and 63 tags change identity,
every existing clone breaks, open PRs need recreating, and 13 sealed receipts become
resolvable only through a translation map. The working tree is identical either way, so
nothing about the code depends on this decision.

If CI clone time or repository size limits are a real constraint, publish with a maintenance
window, the map published alongside, and the migration instructions circulated first. If they
are not, the local rewrite has already delivered its only material benefit — a smaller local
`.git` — and the remote can stay as it is indefinitely.

**This decision requires explicit authorization. It has not been taken.**
