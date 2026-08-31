"""GpuPark must speak the lane lock's protocol, not a second one.

The lane lock is mkdir-atomic: a DIRECTORY with pid/owner files inside, which is
what tools/gpu_lane_lock.sh and every other holder in this repo uses. GpuPark
used touch() + flock on the same path. Two incompatible protocols on one path,
so whichever ran first broke the other - shell first gave the park [Errno 21] Is
a directory (the 30m run hit exactly this), and park first created the 0-byte
regular file that mkdir can never succeed against, which is the wedge seen at
05:58 and cleared at 06:21.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.future import model_bearing_torture as mbt


def test_acquire_creates_a_directory_not_a_file(tmp_path):
    lock = tmp_path / "lane.lock"
    rec = mbt._mkdir_lock_acquire(lock, owner="test")
    try:
        assert lock.is_dir(), "a regular file here is the wedge, not a lock"
        assert (lock / "pid").read_text().strip() == str(os.getpid())
        assert (lock / "owner").read_text().strip() == "test"
        assert rec["protocol"] == "mkdir" and rec["parked"] is False
    finally:
        mbt._mkdir_lock_release(lock)
    assert not lock.exists()


def test_a_wedge_is_reclaimed(tmp_path):
    lock = tmp_path / "lane.lock"
    lock.write_bytes(b"")  # the 0-byte wedge
    mbt._mkdir_lock_acquire(lock, owner="test")
    try:
        assert lock.is_dir()
    finally:
        mbt._mkdir_lock_release(lock)


def test_a_dead_owner_is_reclaimed(tmp_path):
    lock = tmp_path / "lane.lock"
    lock.mkdir()
    (lock / "pid").write_text("999999")   # no such process
    (lock / "owner").write_text("ghost")
    mbt._mkdir_lock_acquire(lock, owner="test")
    try:
        assert (lock / "owner").read_text().strip() == "test"
    finally:
        mbt._mkdir_lock_release(lock)


def test_a_live_holder_is_not_stolen(tmp_path):
    lock = tmp_path / "lane.lock"
    lock.mkdir()
    (lock / "pid").write_text(str(os.getpid()))  # alive
    (lock / "owner").write_text("someone-else")
    with pytest.raises(TimeoutError):
        mbt._mkdir_lock_acquire(lock, owner="test", timeout_s=0.1)
    assert (lock / "owner").read_text().strip() == "someone-else"


def test_release_never_removes_someone_elses_lock(tmp_path):
    lock = tmp_path / "lane.lock"
    lock.mkdir()
    (lock / "pid").write_text("999999")
    (lock / "owner").write_text("someone-else")
    mbt._mkdir_lock_release(lock)
    assert lock.is_dir(), "released a lock this process never held"


def test_the_production_lane_lock_uses_the_mkdir_protocol():
    assert str(mbt.GPU_LOCK) in mbt.MKDIR_PROTOCOL_LOCKS
