# DELEGATION — O004 MODEST GRAVITY (q4-g64; gate profile: MLX/Metal)

Patient O004 = `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (dense; Mistral3ForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a`. Repo: `/Users/scammermike/Downloads/hawking`.
One bounded Gravity candidate (steer S002). SPECIMEN-labelled mlx quant; this
is NOT a Hawking NX win (§15).

Uniform 4-bit group64 (`q4-g64`).

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ tools/doctor_seal.py
READ workspace/campaign/odyssey/patients/O004/census.json
READ workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json
READ receipts/odyssey-i/O004_EXTERNAL.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
Reuse tools/odyssey_patient_runner.py `--gravity q4-g64`. One candidate, do not sweep.
Reload the quantized model, run the SAME fast-Doctor battery + refusal controls.
Measure stored_bytes, stored_bpw (bytes*8/params), active_bytes_per_token + active_bpw
(census active-param split for MoE) and battery/refusal DELTA vs receipts/odyssey-i/O004_EXTERNAL.json.
Write receipts/odyssey-i/O004_GRAVITY_q4-g64.json (schema odyssey.patient.gravity.v1): spec, stored/active bpw, battery,
delta_hits, doctor_seal, SPECIMEN + quant caveat, verdict CANDIDATE_PASS if
delta_hits>=-1 else DEGRADED. Refresh workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json gravity.wins/kills.
Do not delete canonical weights.

## ACCEPTANCE
- receipts/odyssey-i/O004_GRAVITY_q4-g64.json exists with stored_bpw < 16, active_bpw, battery, delta vs baseline,
  SPECIMEN label. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json still schema-valid.

## SCOPE
WRITE tools/odyssey_patient_runner.py
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O004/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, tools/doctor_seal.py, workspace/campaign/odyssey/patients/O004/census.json, workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json, receipts/odyssey-i/O004_EXTERNAL.json
VERIFY tools/odyssey_patient_runner.py by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O004 --weights /Users/scammermike/.cache/huggingface/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a --runtime mlx --out receipts/odyssey-i/O004_GRAVITY_q4-g64.json --packet workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json --skip-route --gravity q4-g64
Do not touch Genesis state or tools/odyssey/.
