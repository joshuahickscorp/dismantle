#!/usr/bin/env bash
# Thin laboratory-harness front-end. Body: tools/bench/archive/path_to_50_verify.sh
# Spec: tools/bench/specs/path_to_50_verify.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export LAB_HARNESS_ARGS="${LAB_HARNESS_ARGS-$*}"
export PYTHONPATH="$ROOT/tools/foundry${PYTHONPATH:+:$PYTHONPATH}"
# Preserve pause/resume lease paths used by historical pipelines.
mkdir -p "$ROOT/artifacts/runs"
exec python3.12 -m lab_harness run "$ROOT/tools/bench/specs/path_to_50_verify.json"
