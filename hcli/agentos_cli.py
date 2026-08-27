"""Command-line surfaces for the provider-neutral AgentOS control plane."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


class _ControlPlaneEngine:
    """Metadata-only engine used by inspection commands; it never generates."""

    def identity(self) -> dict[str, Any]:
        return {"provider": "control-plane", "model_id": "none", "runtime": "inspection"}

    def supports(self, feature: str) -> None:
        del feature
        return None


def _agent(workspace: str, repo_root: Optional[str]) -> Any:
    from hcli.agentos import AgentOS

    return AgentOS(
        Path(workspace).expanduser().resolve(),
        engine=_ControlPlaneEngine(),
        repo_root=Path(repo_root).expanduser().resolve() if repo_root else None,
    )


def _default_repo_root(workspace: str) -> Path:
    for base in (Path.cwd(), Path(__file__).resolve().parents[1]):
        for candidate in (base, *base.parents):
            if (candidate / "hcli").is_dir() and (candidate / "pyproject.toml").is_file():
                return candidate.resolve()
    return Path(workspace).expanduser().resolve()


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=os.getcwd(), help="AgentOS workspace root")
    parser.add_argument("--repo-root", default=None, help="repository root for read/evidence surfaces")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcli agentos",
        description="Inspect and operate the durable provider-neutral AgentOS control plane.",
    )
    sub = parser.add_subparsers(dest="command")

    tools = sub.add_parser("tools", help="list typed tools and their permission contracts")
    _add_paths(tools)
    tools.add_argument("--role", default=None)

    status = sub.add_parser("status", help="show mission, provider, tool, and recovery state")
    _add_paths(status)

    checkpoint = sub.add_parser("checkpoint", help="persist an evidence-backed program checkpoint")
    _add_paths(checkpoint)
    checkpoint.add_argument("--network", action="store_true", help="perform bounded public connectivity probes")
    checkpoint.add_argument("--emit", default=None, help="explicit checkpoint output path")

    recovery = sub.add_parser("recovery-gate", help="run the bounded physical fixture recovery gate")
    recovery.add_argument("--workspace", default=None, help="disposable gate workspace; omitted creates one")
    recovery.add_argument("--emit", default=None)
    recovery.add_argument("--timeout-s", type=float, default=30.0)

    research = sub.add_parser("research-gate", help="run the bounded public research gate")
    research.add_argument("--workspace", default=None)
    research.add_argument("--repo-root", default=None)
    research.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    research.add_argument("--search-query", default="Cloudflare Agents SDK official documentation")
    research.add_argument("--emit", default=None)
    research.add_argument("--timeout-s", type=float, default=12.0)

    vmcp = sub.add_parser("vmcp-gate", help="run the callable VMCP evidence-boundary gate")
    vmcp.add_argument("--workspace", default=None)
    vmcp.add_argument("--repo-root", default=None)
    vmcp.add_argument("--search-query", default="VisionMCP public API evidence tools")
    vmcp.add_argument("--emit", default=None)
    vmcp.add_argument("--timeout-s", type=float, default=12.0)

    native = sub.add_parser("native-gate", help="run the live native HCLI A1-A6 reproduction ladder")
    native.add_argument("--workspace", default=None)
    native.add_argument("--repo-root", default=None)
    native.add_argument("--profile", default=None)
    native.add_argument("--prompt", default="Return exactly: HAWKING_OK")
    native.add_argument("--emit", default=None)
    native.add_argument("--timeout-s", type=float, default=180.0)
    native.add_argument("--model-tokens", type=int, default=64)

    resident = sub.add_parser("resident-gate", help="prove one native resident serves sequential requests")
    resident.add_argument("--workspace", default=None)
    resident.add_argument("--repo-root", default=None)
    resident.add_argument("--profile", default=None)
    resident.add_argument("--count", type=int, default=20)
    resident.add_argument("--timeout-s", type=float, default=180.0)
    resident.add_argument("--model-tokens", type=int, default=32)
    resident.add_argument("--emit", default=None)

    mission_gate = sub.add_parser("native-mission-gate", help="run one live native tool/verifier mission")
    mission_gate.add_argument("--repo-root", default=None)
    mission_gate.add_argument("--profile", default=None)
    mission_gate.add_argument("--emit", default=None)

    background = sub.add_parser("background", help="inspect or manage durable shell-free background jobs")
    bgsub = background.add_subparsers(dest="background_command")
    bg_list = bgsub.add_parser("list", help="list persisted jobs")
    _add_paths(bg_list)
    bg_status = bgsub.add_parser("status", help="inspect one persisted job")
    _add_paths(bg_status)
    bg_status.add_argument("job_id")
    bg_start = bgsub.add_parser("start", help="start argv directly without a shell")
    _add_paths(bg_start)
    bg_start.add_argument("--cwd", default=None)
    bg_start.add_argument("--label", default=None)
    bg_start.add_argument("--timeout-s", type=float, default=None)
    bg_start.add_argument("--non-resumable", action="store_true")
    bg_start.add_argument("argv", nargs=argparse.REMAINDER, help="argv after --")
    bg_resume = bgsub.add_parser("resume", help="rerun an interrupted resumable job")
    _add_paths(bg_resume)
    bg_resume.add_argument("job_id")
    bg_cancel = bgsub.add_parser("cancel", help="cancel one running job")
    _add_paths(bg_cancel)
    bg_cancel.add_argument("job_id")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command is None:
        build_parser().print_help()
        return 0
    try:
        if args.command == "recovery-gate":
            from hcli.agentos.recovery import run_recovery_gate

            report = run_recovery_gate(
                args.workspace,
                emit=args.emit,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "research-gate":
            from hcli.agentos.research import run_research_gate

            report = run_research_gate(
                args.workspace,
                repo_root=args.repo_root,
                repo=args.repo,
                search_query=args.search_query,
                emit=args.emit,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "vmcp-gate":
            from hcli.agentos.vmcp_gate import run_vmcp_gate

            report = run_vmcp_gate(
                args.workspace,
                repo_root=args.repo_root,
                emit=args.emit,
                search_query=args.search_query,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "native-gate":
            from hcli.agentos.native_gate import run_native_gate

            report = run_native_gate(
                args.workspace,
                repo_root=args.repo_root,
                profile=args.profile,
                prompt=args.prompt,
                emit=args.emit,
                timeout_s=args.timeout_s,
                model_tokens=args.model_tokens,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "resident-gate":
            from hcli.agentos.resident_gate import run_resident_gate

            report = run_resident_gate(
                args.workspace,
                repo_root=args.repo_root,
                profile=args.profile,
                count=args.count,
                timeout_s=args.timeout_s,
                model_tokens=args.model_tokens,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "native-mission-gate":
            from hcli.agentos.native_mission_gate import run_native_mission_gate

            report = run_native_mission_gate(
                repo_root=args.repo_root,
                profile=args.profile,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "checkpoint":
            from hcli.agentos.checkpoint import write_program_checkpoint

            report = write_program_checkpoint(
                args.repo_root or _default_repo_root(args.workspace),
                workspace=args.workspace,
                emit=args.emit,
                network=args.network,
            )
            _emit(report)
            return 0

        agent = _agent(args.workspace, args.repo_root)
        if args.command == "tools":
            _emit({"schema": "hcli.agentos.tool_catalog.v1", "tools": agent.tools.discover(role=args.role)})
            return 0
        if args.command == "status":
            _emit(agent.status())
            return 0
        if args.command == "background":
            command = args.background_command
            if command == "list":
                _emit({"schema": "hcli.agentos.background_catalog.v1", "jobs": agent.background_jobs()})
                return 0
            if command == "status":
                _emit(agent.background_status(args.job_id))
                return 0
            if command == "start":
                argv_value = list(args.argv)
                if argv_value and argv_value[0] == "--":
                    argv_value = argv_value[1:]
                if not argv_value:
                    raise ValueError("background start requires argv after --")
                _emit(agent.start_background(
                    argv_value,
                    cwd=args.cwd,
                    label=args.label,
                    resumable=not args.non_resumable,
                    timeout_s=args.timeout_s,
                ))
                return 0
            if command == "resume":
                _emit(agent.resume_background(args.job_id))
                return 0
            if command == "cancel":
                _emit(agent.cancel_background(args.job_id))
                return 0
            build_parser().parse_args(["background", "--help"])
            return 0
    except Exception as exc:  # CLI surfaces must leave a machine-readable failure.
        _emit({
            "schema": "hcli.agentos.cli_error.v1",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        return 1
    return 0


__all__ = ["build_parser", "main"]
