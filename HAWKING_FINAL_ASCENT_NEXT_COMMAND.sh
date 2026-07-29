#!/usr/bin/env bash
# Historical final-ascent resume shell. The generator
# tools/campaign/final_ascent_status.py was product-released under C-HIST-R1.
# Default mode is READ-ONLY diagnose. Explicit action flags required for state changes.
# Stale leases must be refused; MOP preserved.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${1:-diagnose}"

refuse_stale_lease() {
  if [[ -n "${HAWKING_LEASE_ID:-}" ]]; then
    echo "REFUSING action under HAWKING_LEASE_ID=$HAWKING_LEASE_ID without explicit --accept-lease" >&2
    exit 2
  fi
}

diagnose() {
  echo "=== HAWKING FINAL ASCENT diagnose (read-only) ==="
  echo "endpoint: RAMANUJAN_SANDBOX_READY reached=False"
  echo "fences: ODYSSEY_LAUNCH_AUTHORIZED=False RAMANUJAN_RESEARCH_AUTHORIZED=False HIDE_KERNEL_TURN=False"
  echo "capability_gate: REFUSED any_approved=False"
  echo "generation_b: REFUSED"
  echo "q0_achieved: True"
  echo
  echo "=== git ==="
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  echo
  echo "=== fences on disk ==="
  if [[ -f odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED ]]; then
    echo -n "ODYSSEY_LAUNCH_AUTHORIZED="; cat odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED; echo
  else
    echo "ODYSSEY_LAUNCH_AUTHORIZED file absent (treat as false)"
  fi
  echo
  echo "=== hawking launchd (observe only) ==="
  launchctl list 2>/dev/null | grep com.hawking || true
  echo
  echo "=== safe next commands (not executed) ==="
  echo "  # C-HIST-R1 released: tools/campaign/final_ascent_status.py, light_governor.py"
  echo "  # Rebuild apparatus still live:"
  echo "  python3.12 tools/campaign/rung_gate.py --help"
  echo "  python3.12 tools/campaign/ledger_rollup.py --help"
  echo "  python3.12 -m lab --classify"
  echo "  cat GLM52_GENERATION_B_CAPABILITY_VERDICT.json"
  echo "  cat odyssey/launch/SUBSTRATE_CAPABILITY.json"
  echo '  # FA02 blocking: produce a capable Math-Preserve-v2, then:'
  echo '  .venv/glm52/bin/python tools/condense/glm52_capability_gate.py \'
  echo '    --artifact PATH/TO/PACKED_DIR --run --out CAPABILITY.json'
  echo '  # Only after G_math+G_live PASS may the controller bind the hash APPROVED.'
  echo
  echo "MOP: not inspected beyond process-name avoidance; do not touch."
}

reconcile() {
  diagnose
  echo
  echo "=== reconcile: historical generator released under C-HIST-R1 ==="
  echo "REFUSED: tools/campaign/final_ascent_status.py is product-released (not invocable)."
  echo "Sealed status artifacts remain readable; do not invent a replacement generator."
  exit 4
}

action_help() {
  echo "Action mode is explicit. Supported:"
  echo "  $0 diagnose     # default, read-only"
  echo "  $0 reconcile    # refused: historical generator product-released (C-HIST-R1)"
  echo "  $0 action ...   # refused unless a future explicit allowlist is added"
  echo "Never: kill/start launch agents, flip fences, touch MOP, delete capsules."
}

case "$MODE" in
  diagnose|--diagnose|ro|read-only) diagnose ;;
  reconcile|--reconcile) refuse_stale_lease; reconcile ;;
  action|--action)
    refuse_stale_lease
    echo "REFUSED: no destructive/live action allowlist is enabled in this revision." >&2
    action_help
    exit 3
    ;;
  -h|--help) action_help ;;
  *)
    echo "unknown mode: $MODE" >&2
    action_help
    exit 2
    ;;
esac
