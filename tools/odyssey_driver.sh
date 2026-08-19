#!/usr/bin/env bash
# Odyssey-I one-tick driver: reap finished lanes, then launch next (self-gated).
# Launchd template: workspace/campaign/odyssey/com.hawking.odyssey.plist
# Do not load that plist from this script.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ODYSSEY_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3}"
if [ ! -x "$PY" ]; then
  PY=python3
fi

LOG="$ROOT/workspace/campaign/odyssey/driver.log"
mkdir -p "$(dirname "$LOG")"
ts="$(date '+%Y-%m-%dT%H:%M:%S%z')"

{
  echo "== odyssey driver tick $ts =="
  echo "repo=$ROOT py=$PY"
  ODYSSEY_HEADROOM_ADMIT=1 "$PY" tools/odyssey_ctl.py cycle --go --max-lanes 2
  echo "-- cycle rc=$? --"
  echo "== tick done $ts =="
} >>"$LOG" 2>&1
