"""The GPU lane lock must reclaim a non-directory at its path.

A 0-byte regular file appeared at /tmp/hawking-gpu-lane.lock twice on this box:
at 05:58 (cleared 06:21) and again at 08:41. mkdir can never succeed against an
existing regular file, and the stale-owner branch tests "$LOCK/pid" - which is
unreachable when $LOCK is a file - so the rm -rf that clears a dead owner could
never fire. Every caller spun to its 5400-second deadline and exited 75, meaning
GPU lanes were either timing out or running UNSERIALISED, and any absolute GB/s
measured in such a window has no serialisation guarantee.

Nothing in this repo creates it as a file; every reader and writer here uses
mkdir / create_dir / [ -d ]. Rather than keep hunting the creator, the lock
reclaims the state.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SH = REPO / "tools" / "future" / "test_gpu_lane_lock_wedge.sh"


def test_wedge_is_reclaimed_live_holder_is_respected():
    r = subprocess.run([str(SH)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "wedge reclaimed" in r.stdout


def test_the_script_still_uses_a_directory_lock():
    """The fix must not quietly turn the lock into a file lock, which would make
    two lanes think they both hold it."""
    src = (REPO / "tools" / "gpu_lane_lock.sh").read_text()
    assert 'mkdir "$LOCK"' in src, "acquisition must stay mkdir-atomic"
    assert 'rm -f "$LOCK"' in src, "the wedge branch must remove the file"
    assert '[ ! -d "$LOCK" ]' in src, "the wedge branch must test for non-directory"
