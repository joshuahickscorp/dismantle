#!/usr/bin/env bash
# Odyssey-I RESIDENT driver: loops continuously — reap finished lanes, launch the
# next rungs, keep the box working. Not a 15-min cron burst; a resident that ticks
# every TICK_SECS so there is (almost) always a model runner PID in flight while
# the deterministic descent-ladder well has work.
# Launchd template (KeepAlive): workspace/campaign/odyssey/com.hawking.odyssey.plist
# Do not load that plist from this script.
set -uo pipefail   # not -e: one bad tick must never kill the resident

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# launchd runs with a minimal environment: restore HOME + a full PATH so git/date/
# python3 resolve. (grok is intentionally NOT in the loop, but keep PATH complete.)
export HOME="${HOME:-/Users/scammermike}"
export PATH="$HOME/.grok/bin:$HOME/.claude-grok/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

PY="${ODYSSEY_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3}"
if [ ! -x "$PY" ]; then
  PY=python3
fi

LOG="$ROOT/workspace/campaign/odyssey/driver.log"
mkdir -p "$(dirname "$LOG")"

TICK_SECS="${ODYSSEY_TICK_SECS:-12}"     # cadence between reap+launch ticks
COMMIT_EVERY="${ODYSSEY_COMMIT_EVERY:-25}"  # commit data every N ticks (~5 min)
# Single-resident lock: if another resident holds it, exit (launchd KeepAlive keeps one).
LOCK="$ROOT/workspace/campaign/odyssey/.resident.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  # stale lock if owner PID is gone
  if [ -f "$LOCK/pid" ] && ! kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then
    rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || exit 0
  else
    echo "$(date '+%FT%T%z') resident already running; exit" >>"$LOG"; exit 0
  fi
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM

commit_data() {
  git add -- receipts/odyssey-i \
    workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json \
    workspace/campaign/odyssey/ODYSSEY_STATE.json \
    workspace/campaign/odyssey/RUN_LOG.jsonl \
    workspace/campaign/odyssey/TRANSFER_MATRIX.json \
    workspace/campaign/odyssey/GRAVITY_RULEBASE.json \
    workspace/campaign/odyssey/NEGATIVE_SCIENCE.json \
    workspace/campaign/odyssey/patients >/dev/null 2>&1 || true
  git commit -q -m "odyssey-i resident: autonomous cycle $(date '+%FT%T%z')" >/dev/null 2>&1 \
    && echo "-- committed data --" || true
}

echo "== odyssey RESIDENT up $(date '+%FT%T%z') pid=$$ tick=${TICK_SECS}s ==" >>"$LOG"
tick=0
while true; do
  tick=$((tick+1))
  ts="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  {
    echo "== tick $tick $ts =="
    # --max-lanes = MODEL concurrency ceiling (memgate is the real limiter);
    # --grok-lanes 0 = NO grok in the loop (deterministic science only).
    ODYSSEY_HEADROOM_ADMIT=1 "$PY" tools/odyssey_ctl.py cycle --go --max-lanes 12 --grok-lanes 0
    echo "-- cycle rc=$? --"
  } >>"$LOG" 2>&1
  if [ $((tick % COMMIT_EVERY)) -eq 0 ]; then
    commit_data >>"$LOG" 2>&1
  fi
  sleep "$TICK_SECS"
done
