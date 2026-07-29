#!/bin/sh
# Historical continuum resume point.
# tools/campaign/continuum_status.py was already absent; C-HIST-R1 does not restore it.
# state: SEAL_CLAIM_A (historical)
cd "$(dirname "$0")"
echo "C-HIST-R1: continuum_status generator is not invocable (historical / already absent)."
echo "Use rebuild apparatus instead:"
echo "  python3.12 tools/campaign/rung_gate.py --help"
echo "  python3.12 tools/campaign/ledger_rollup.py --help"
echo "  python3.12 -m lab --classify"
