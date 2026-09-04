# HCLI bootstrap record

Dated snapshots of HCLI's bootstrap form and the plans written around it. This
directory is a **record, not code**. Nothing imports it, nothing runs it, and
its contents are deliberately unedited.

## Why the files still say "haider"

HCLI's bootstrap form was called HAIDER — a contraction of *aider*, the
third-party tool that was its temporary substrate. That dependency is gone
(zero live imports, audited in `receipts/future/AIDER_NAMESPACE_AUDIT.json`),
and on 2026-09-02 the name went with it: the tests moved to `hcli/tests/`, the
Rust module became `hide_backend::hcli`, `parse_haider_args` became
`parse_hcli_args`, and the verbatim upstream `CoderPrompts` source that was
still checked in was deleted.

The dated Python snapshots and pre-change wrappers were retired in the Event
Horizon compaction. They remain recoverable in Git history, while this
directory keeps the plans and the explanation of the bootstrap era. Renaming
or rewriting those fossils in place would have made the record say something
that never happened — the same reason `receipts/` was left alone.

`hcli/tests/test_module_identity.py` keeps the fossil dotted name
`tools.haider.hcli.engine` alive for the same reason: it asserts that name does
NOT resolve, and it is built so a mechanical rename cannot delete the very
string it is guarding against.

## What is here

- `P0_*.md`, `P1_*.md`, `SELF_HOST_GROUND_EDIT.md` — the plans of that period
- the deleted Python snapshots and wrappers remain available from Git history

## What is not here

`aider_patches/` held a verbatim copy of upstream aider's `CoderPrompts` class
and a patch against it. Nothing read either one. Both were deleted rather than
moved: third-party source is not our record to keep.
