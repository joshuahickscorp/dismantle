"""Every Hawking process gets a role, and a wrong role is worse than none.

The argv shapes below are the real ones observed on this host on 2026-09-01, not
invented fixtures. The one that matters most is the near-miss: a 3 MB process
carrying `--artifact-root` was labelled `resident-body` beside the actual 1.18 GB
body, which is exactly the kind of confident-but-wrong label this view exists to
replace.
"""
from __future__ import annotations

from hcli.processes import _body_of, _classify, live_processes, render, summary

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




def test_memory_comes_from_footprint_not_rss():
    """RSS is not Activity Monitor's Memory column, and here it is not close.

    Measured on this host: the resident body reported rss=1.19 GB against a
    phys_footprint of 12 GB, and two `hf download` children reported 2.93 and
    1.73 GB against 29.84 GB each in Activity Monitor. RSS counts resident pages
    and ignores what the compressor holds for the process; with tens of GB
    compressed system-wide that is most of the footprint. Reporting RSS made a
    box under real memory pressure look idle, which is the one thing this view
    exists to prevent.
    """
    import os

    from hcli.processes import _footprint_bytes

    procs = live_processes()
    if not procs:
        return  # nothing running is a valid host state, not a failure

    # THE ASSERTION THAT BITES. An earlier version of this test accepted either
    # source, so reverting the fix to `measured = None` left it green while the
    # view silently went back to under-reporting by ~10x. Where the platform CAN
    # give a footprint, the view MUST use it; hosts without the tool still skip.
    available = _footprint_bytes(os.getpid())
    if available is not None:
        assert any(p.memory_source == "phys_footprint" for p in procs), (
            "footprint is available on this host but every process fell back to "
            "rss, which under-reports memory by roughly ten-fold"
        )
    sources = {p.memory_source for p in procs}
    assert sources <= {"phys_footprint", "rss"}, sources
    # Every process must SAY which metric it used, so a silent fallback to the
    # under-reporting one can be seen rather than believed.
    for proc in procs:
        assert proc.memory_source in ("phys_footprint", "rss")
        assert proc.to_dict()["memory_source"] == proc.memory_source


def test_the_fallback_is_labelled_in_the_rendered_view():
    """An rss fallback must announce itself; a quiet one is the old bug back."""
    from hcli.processes import Process

    fell_back = [Process(
        pid=1, ppid=0, rss_bytes=1234567, cpu_percent=0.0, elapsed="1:00",
        role="resident-body", process_class="ESSENTIAL_PERSISTENT",
        safe_to_stop=False, purpose="p", command="c", memory_source="rss",
    )]
    text = render(fell_back, width=200)
    assert "rss fallback" in text and "under-reports" in text, text

    measured = [Process(
        pid=1, ppid=0, rss_bytes=1234567, cpu_percent=0.0, elapsed="1:00",
        role="resident-body", process_class="ESSENTIAL_PERSISTENT",
        safe_to_stop=False, purpose="p", command="c",
        memory_source="phys_footprint",
    )]
    assert "rss fallback" not in render(measured, width=200)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all green")
