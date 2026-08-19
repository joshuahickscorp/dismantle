# HAWKING ODYSSEY-I — ACTIVE LEDGER

Prime directive: **Odyssey-I perfects HAWKING, not the models.** Every patient is a
teacher; every patient must leave the compiler more capable. Product = Gravity /
Doctor / NX / NVM-HIDE / Machine-Genome vNEXT + rulebase + transfer matrix +
negative-science corpus. (Full doctrine: `/Users/scammermike/Downloads/h_odyssey.md`.)

Compact by law (§20). Closed obligations get archived out of active nomenclature.

## Operating rules (enforced)
- **Cheapest reliable layer** (§10): script > grok > local model > Opus. Move work down.
- **Opus detached** (§8): Opus = chief scientist / arbiter / adversarial judge only.
  Escalate on the §9 triggers (contradiction, breakthrough, false-win, retirement,
  arch-first, rule-promotion, resource-fork, phase-synthesis). Not for polling/counting.
- **Determinism first** (§13): tools MEASURE, models interpret. `tools/odyssey_census.py`.
- **Protected GPU** (§14): one authoritative timing lane; timing under load = VOID.
- **Memory-safety gate** (§15): admit local workers via `worker_gate`, not naive RSS.
- **Disk is a cache** (§16): active window = current + next patient + compiler cache.
  Seal science (receipts/provenance/packet/rules) before deleting any bulk body.
- **Receipt epistemology** (§18): every result labelled VERIFIED/MEASURED/DERIVED/
  INFERRED/HYPOTHESIS/SPECULATIVE/REFUTED/STALE/UNKNOWN. Never promote projection→measurement.
- **False-win gates** (§60): effective-BPW, active-byte, expansion, foreign-runtime,
  fallback, content-provenance, doctor-control, gpu-contamination, capture-adequacy, gain.
- **Never stop on an ordinary result** (§98). Harvest, record, refill the queue.

## Patient queue  (state · phase)
Phases: INGEST→BASELINE→CENSUS→ROUTEMAP→SENSITIVITY→GRAVITY→NX→KERNEL→DOCTOR→PACKET→TRANSFER→SEAL

| Oxx | model | class | source | gate | state | phase |
|-----|-------|-------|--------|------|-------|-------|
| O000 | gemma-3-1b-it | tiny dense (compiler lab) | google/gemma-3-1b-it | **HF-gated** | BLOCKED-auth | — |
| O001 | Falcon-H1-7B-Instruct | small hybrid attn+Mamba | tiiuae/Falcon-H1-7B-Instruct | open | **on-disk** | CENSUS✓ |
| O002 | gemma-3-4b-it | small dense multimodal | google/gemma-3-4b-it | **HF-gated** | BLOCKED-auth | — |
| O003 | Kimi-VL-A3B-Instruct | small multimodal MoE | moonshotai/Kimi-VL-A3B-Instruct | open | queued | — |
| O004 | Mistral-Small-3.1-24B | medium dense multimodal | mistralai/Mistral-Small-3.1-24B-Instruct-2503 | **HF-gated** | BLOCKED-auth | — |
| O005 | Qwen3-30B-A3B | small-active MoE | Qwen/Qwen3-30B-A3B | open | **on-disk** | CENSUS✓ |
| O006 | Qwen3-VL-30B-A3B | multimodal MoE sibling | Qwen/Qwen3-VL-30B-A3B-Instruct | open | queued (transfer ctrl) | — |
| O007 | Kimi-Linear-48B-A3B | linear-attn MoE (KDA+MLA) | moonshotai/Kimi-Linear-48B-A3B-Instruct | open | queued | — |
| O008 | Jamba-Mini-1.5 | Mamba+attn MoE | ai21labs/AI21-Jamba-Mini-1.5 | license? | queued | — |
| O009 | Qwen2.5-72B-Instruct | large dense | Qwen/Qwen2.5-72B-Instruct | open | queued (large) | — |
| O010 | GLM-4.5-Air | 106B/12B MoE | zai-org/GLM-4.5-Air | open | queued (~220GB) | — |
| O011 | DSV4F | **mandatory legacy replay** | reconstruct from receipts | local | queued (control) | — |
| O012 | GLM-4.5 full | 355B/32B very-large MoE | zai-org/GLM-4.5 | open | queued (partial-residency) | — |
| O013 | Kimi-K3 | frontier 2.8T/104B streamed | moonshotai/Kimi-K3 | open | queued (streamed capstone) | — |

Scheduler (§50): numbers = scientific ladder; run cheapest-actual order. Backfill small
patients while large downloads run. O005/O010 were the "already in motion" pair.

## Active obligations  (small; §20)
- **A1** O005 route/state map: instrument router, measure expert freq / entropy / P(E_t|E_t-1) / co-occurrence (§57). READY.
- **A2** O005 baseline TPS via external runtime (MLX/transformers) — the generic-runtime reference. READY (needs runtime pick from reuse recon).
- **A3** O005 sensitivity map: per-organ / per-expert Doctor sensitivity (experts = 95% of body, 11% active). READY.
- **A4** Control plane: patient controller + harvester + packet builder + rulebase/transfer/negative seeds → delegate to Grok once reuse-surface recon lands.
- **A5** O001 hybrid census refinement: add SSM organ bucket; measure state bytes vs KV (§36 long-context). READY.
- **A6** HF-token: unblock O000/O002/O004 (gated). NEEDS USER.

## Compiler frontier
- Rulebase:        `GRAVITY_RULEBASE.json`   (seed from Qwen3.8/Q80/DSV4F science) — TODO A4
- Transfer matrix: `TRANSFER_MATRIX.json`    (rule × patient) — TODO A4
- Negative corpus: `NEGATIVE_SCIENCE.json`   (killed mechanisms + reopen_if) — TODO A4
- Machine genome:  M3 Ultra 96GB (ambient, authoritative) — improve per patient (§66)

## Metrics (§72)  — session bootstrap
opus_calls≈1 · grok_lanes=3 (arch-archaeology, reuse-surface, +downloads) · gpu_seconds=0
patients_on_disk=2 (O001,O005) · patients_censused=2 · universal_rules=0(seed pending)

## Measured so far (MEASURED)
- O005 Qwen3-30B-A3B: 30.532B total / **3.353B active-per-token (11.0%)**; stored 61.06GB bf16;
  experts=57.98GB (95% of body), attn=1.81, router=0.03, embed/lm_head=0.62 ea. → experts are the lever.
- O001 Falcon-H1-7B: 7.586B, 44 layers, 256K ctx, hybrid attn(12/2kv)+Mamba(d_state256,dconv4,24heads);
  SSM machinery 2.64GB, mlp 9.97GB, attn 0.97GB.
