"""Every Hawking process gets a role, and a wrong role is worse than none.

The argv shapes below are the real ones observed on this host on 2026-09-01, not
invented fixtures. The one that matters most is the near-miss: a 3 MB process
carrying `--artifact-root` was labelled `resident-body` beside the actual 1.18 GB
body, which is exactly the kind of confident-but-wrong label this view exists to
replace.
"""
from __future__ import annotations

from hcli.processes import _body_of, _classify, render, summary

RESIDENT_BODY = (
    "/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/release-fast/"
    "examples/ascension_qwen38_resident --artifact-root "
    "/Users/scammermike/noetic/NOETIC_PARENT_A --tokenizer "
    "/Users/scammermike/noetic/NOETIC_PARENT_A/tokenizer.json --max-seq-len 8192 "
    "--resident-identity sealed-3.14"
)
SUPERVISOR = (
    "/opt/homebrew/.../Python -m hcli.agentos.resident --supervise "
    "/tmp/ws/.hcli/resident/state.json"
)
SOVEREIGN = (
    "/Users/scammermike/.venvs/hawking-aider/bin/python3 "
    "tools/future/hcli_sovereign.py --run --minutes 600"
)
WATCHER = "/Library/.../Python /Users/scammermike/Downloads/hawking/tools/odyssey/modellake_watch.py --poll-secs 0.10"
DOWNLOAD = (
    "/Library/.../Python /Library/.../bin/hf download Qwen/Qwen3-Coder-30B-A3B-Instruct "
    "config.json --revision b2cff646eb4b --local-dir /Volumes/corpdrive/x --max-workers 16"
)


def _role(command: str):
    meta = _classify(command)
    return None if meta is None else meta[0]


def test_each_real_process_shape_gets_its_role():
    assert _role(RESIDENT_BODY) == "resident-body"
    assert _role(SUPERVISOR) == "resident-supervisor"
    assert _role(SOVEREIGN) == "sovereign-loop"
    assert _role(WATCHER) == "modellake-watcher"
    assert _role(DOWNLOAD) == "modellake-download"


def test_a_flag_alone_does_not_make_something_the_resident_body():
    """The near-miss that shipped a wrong label once already."""
    probe = "/Library/.../Python tools/accelerator/some_probe.py --artifact-root /x/y"
    assert _role(probe) != "resident-body"
    gate = "/bin/sh -c 'run_gate --artifact-root /x/y --resident-identity sealed-3.14'"
    # A shell wrapper is not the body either; the executable must be a resident.
    assert _role(gate) != "resident-body"


def test_unrelated_processes_are_not_claimed():
    for command in (
        "/usr/bin/python3 -m pip install requests",
        "/Applications/Safari.app/Contents/MacOS/Safari",
        "node /Users/x/.claude-grok/v2/contract.mjs",
        "",
    ):
        assert _classify(command) is None, command


def test_the_downloader_is_the_only_safe_to_stop_persistent_thing():
    """Stop-safety is the field an operator acts on, so it must not be sloppy."""
    assert _classify(DOWNLOAD)[2] is True, "a resumable download is safe to stop"
    for command in (RESIDENT_BODY, SUPERVISOR, SOVEREIGN, WATCHER):
        assert _classify(command)[2] is False, command


def test_model_identity_is_read_as_data_not_from_the_executable_name():
    """The body's identity comes off argv. The binary's name is not the truth."""
    assert _body_of(RESIDENT_BODY) == "sealed-3.14"
    assert _body_of(DOWNLOAD) == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert _body_of(SUPERVISOR) is None


def test_summary_and_render_survive_a_host_with_nothing_running():
    """Called on any machine, including one where Hawking is not running."""
    assert render([]) == "no Hawking processes visible on this host"
    live = summary()
    assert set(live) >= {"count", "total_rss_bytes", "by_class", "processes"}
    assert live["count"] == len(live["processes"])


def test_render_never_exceeds_its_width():
    text = render(width=60)
    assert all(len(line) <= 60 for line in text.splitlines()), text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all green")
