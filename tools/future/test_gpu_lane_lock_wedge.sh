#!/bin/bash
# The GPU lane lock must reclaim a non-directory at its path.
#
# A 0-byte regular file appeared at /tmp/hawking-gpu-lane.lock twice on this box
# (05:58, cleared 06:21; again at 08:41). mkdir can never succeed against an
# existing file, and the stale-owner branch tests "$LOCK/pid" - unreachable when
# $LOCK is a file - so the rm -rf that clears a dead owner could never fire.
# Every caller spun to its 5400s deadline and exited 75, which means GPU lanes
# were either timing out or running unserialised.
set -u
LOCK_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tools/gpu_lane_lock.sh"
fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. a wedge (regular file) is reclaimed and the command still runs
rm -rf /tmp/hawking-gpu-lane.lock
: > /tmp/hawking-gpu-lane.lock
out="$("$LOCK_SCRIPT" wedgetest /bin/echo ran 2>/dev/null)" || fail "wedge not reclaimed"
[ "$out" = "ran" ] || fail "command did not run after reclaim (got: $out)"

# 2. the lock is released afterwards
[ -e /tmp/hawking-gpu-lane.lock ] && fail "lock not released after exit"

# 3. a LIVE holder is still respected - the wedge branch must not steal a real lock
mkdir /tmp/hawking-gpu-lane.lock
echo $$ > /tmp/hawking-gpu-lane.lock/pid
echo livetest > /tmp/hawking-gpu-lane.lock/owner
timeout 8 "$LOCK_SCRIPT" intruder /bin/echo stolen >/dev/null 2>&1 && fail "a live lock was stolen"
[ -d /tmp/hawking-gpu-lane.lock ] || fail "live lock was destroyed"
rm -rf /tmp/hawking-gpu-lane.lock

echo "ok: wedge reclaimed, lock released, live holder respected"
