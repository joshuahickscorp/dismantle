# Reaching 300k requires a product decision, not more engineering

Current: **430,406**. Target: **300,000**. Gap: **130,406**.

```
hawking       170,792   39.7%
laboratory    129,091   30.0%
hide          107,871   25.1%
shared         22,652    5.3%
```

Every condensation mechanism the campaign names has been attempted and measured. Four
architectures were refuted, seven subsystems were attacked, a destructive probe deleted every
protected item, and a direct lane condensed the last untouched trees. The engineering answer
is exhausted; what remains are three product decisions, each with a measured LOC consequence.

---

## Option A — relocate HIDE to a sibling repository

**Frees: 107,871 (25.1%). Leaves: 322,535.**

Does not reach 300,000 on its own, and **earns zero credit under this campaign's rules**,
which count relocation separately from elimination. It would, however, make the Hawking
repository materially smaller for anyone working on inference.

What it costs: a versioned Hawking/HIDE ABI, a second release pipeline, and the shared-session
behaviour tests the campaign specifies. The `hawking` binary already depends on zero hide
crates, so the seam is clean — this is the cheapest of the three to execute.

What it does not solve: the combined sum is unchanged by definition.

---

## Option B — retire live campaign machinery

**Frees: up to 129,091 (30.0%). Leaves: 301,315 if taken in full.**

This is the only option that reaches the target alone, and it is the most expensive in
capability. The laboratory is not dead code — a lane measured its irreducible residue at
29,541 lines of Odyssey T0 and its sealed negatives, the Math-Preserve prometheus passes,
ramanujan, and the `final_ascent` and `light_governor` publishers.

Retiring it means accepting that a sealed campaign cannot be re-run from its own machinery,
only from its specification, fixture and receipt. The campaign document explicitly protects
negative science, so this option contradicts one of its own rules and needs you to override it
deliberately rather than have me infer it.

---

## Option C — cut HIDE product surface

**Frees: an unmeasured fraction of 107,871.**

A lane with full scope — twenty crates, the frontend, the memory substrate and the workspace
manifest — returned 2,536 lines, 8% of its target, and named the residue as fleet fabric, YOU,
personalize, compat, program-runtime and the backend host integration surface. Going further
means removing shipped product capability, not duplication.

This is a decision about what HIDE is, not about how it is written.

---

## What I recommend

**Do not chase 300,000.** It was set before the irreducible budget was measured, and the
measurement says the number costs either negative science or shipped product.

The arc already delivered what a condensation campaign can honestly deliver: **172,263 lines
eliminated, 28.6%, none relocated, with the logical assertion inventory up from 3,891 to 3,925
and zero capabilities lost.** The repeated architecture the campaign existed to remove is gone
— one campaign engine replaced fourteen controllers, one laboratory harness replaced four
frameworks, and the test suites were compressed without losing an assertion.

If a smaller inference repository is the actual goal rather than a smaller number, **Option A**
achieves it at the lowest cost, and the campaign's zero-credit rule for relocation is what
makes that invisible in the headline figure.

---

## Rollback

Every rung is tagged. `git checkout <tag>` restores it.

```
arc-rung-550k-green    536,532
arc-rung-475k-green    458,272
arc-447k-green         447,743
arc-437k-green         437,199
arc-430k-green         430,406
```

Rewritten history remains local. `docs/COMMIT_MAP_20260728.txt` holds 1,819 old-to-new pairs,
both backup bundles are under `~/Downloads/hawking-backup/`, and
`HISTORY_PUBLICATION_PACKET.md` documents the publication impact. Nothing has been
force-pushed.
