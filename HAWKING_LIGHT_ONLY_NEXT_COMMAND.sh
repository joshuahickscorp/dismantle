#!/usr/bin/env bash
# Re-derive light-only campaign state. Safe to run at any time; changes nothing.
# C-HIST-R1 product-released tools/campaign/light_governor.py (and historical light_status).
set -euo pipefail
cd "$(dirname "$0")"
echo "C-HIST-R1: tools/campaign/light_governor.py is product-released (not invocable)."
echo "Rebuild apparatus still live:"
echo "  python3.12 tools/campaign/rung_gate.py --help"
echo "  python3.12 tools/campaign/ledger_rollup.py --help"
echo "  python3.12 -m lab --classify"
echo
echo 'Historical note: HEAVY_WINDOW_AVAILABLE was a light_governor signal.'
echo 'It is never started automatically. A human decides.'
