# Campaign / experiment engine (lane A1)

One authority for campaign lifecycle. Controllers do not multiply.

## Surfaces

| Module | Role |
|--------|------|
| `spec.py` | `ExperimentSpec` schema — campaigns are data |
| `state_machine.py` | Typed FSM for precheck → … → report |
| `scheduler.py` | Resume-safe work planning |
| `governor.py` | Disk / resource floors |
| `lease.py` | Exclusive controller lease |
| `checkpoint.py` | Hash-chained events + sealed checkpoints |
| `receipts.py` | Canonical sealed receipts |
| `seal_integrity.py` | Reseal/substitution rejection, launcher-node safety, no-subprocess preflight |
| `runtime.py` | `run_campaign` / CLI |

## Specs

Declarative JSON under `specs/` for:

- `glm52` (live contract readers remain as modules)
- `kimi_k26`, `qwen`, `gptoss`, `deepseek_v4`, `second_light`, `gravity_frontier`

Each keeps: steps, receipt pointer, fixture, reproduction command, reopen conditions.

## Run

```bash
python3.12 -m tools.condense.engine tools/condense/engine/specs/glm52.json \
  --work-dir reports/condense/engine/glm52
```

Retired controller bodies live under `tools/condense/archive/` (excluded from active LOC).
Public import paths stay as thin shims that exec the archived source so tests and
monkeypatches keep working.
