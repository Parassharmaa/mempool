from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mempool.terminal_bench import summarize_harbor_job


Runner = Callable[..., subprocess.CompletedProcess[str]]


def build_harbor_preflight_command(
    task_path: Path,
    job_name: str,
    jobs_dir: Path,
    *,
    agent: str = "oracle",
    model: str | None = None,
    agent_kwargs: Sequence[str] = (),
    allow_agent_hosts: Sequence[str] = (),
    install_only: bool = False,
    n_attempts: int = 1,
    n_concurrent: int = 1,
) -> list[str]:
    command = [
        "uvx",
        "harbor",
        "run",
        "--path",
        str(task_path),
        "--agent",
        agent,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        str(n_attempts),
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
        "--yes",
    ]
    if model:
        command.extend(["--model", model])
    for agent_kwarg in agent_kwargs:
        command.extend(["--agent-kwarg", agent_kwarg])
    for host in allow_agent_hosts:
        command.extend(["--allow-agent-host", host])
    if install_only:
        command.append("--install-only")
    return command


def run_harbor_preflight(
    command: Sequence[str],
    job_dir: Path,
    output: Path,
    *,
    timeout_seconds: int,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    started = time.monotonic()
    process_status = "not_started"
    returncode: int | None = None
    error: str | None = None
    try:
        completed = runner(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
        process_status = "exited"
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        process_status = "timeout"
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        process_status = "error"
        error = f"{type(exc).__name__}: {exc}"

    elapsed_seconds = round(time.monotonic() - started, 3)
    payload: dict[str, Any] = {
        "command": list(command),
        "process_status": process_status,
        "returncode": returncode,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": elapsed_seconds,
        "raw_log_policy": "not_read",
    }
    if error:
        payload["error"] = error

    if (job_dir / "result.json").exists():
        payload["harbor_summary"] = summarize_harbor_job(job_dir)
    else:
        payload["harbor_summary"] = None

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a timeout-bounded Harbor Terminal-Bench preflight and write a safe summary."
    )
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", default="oracle")
    parser.add_argument("--model")
    parser.add_argument("--agent-kwarg", action="append", default=[])
    parser.add_argument("--allow-agent-host", action="append", default=[])
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    command = build_harbor_preflight_command(
        args.task_path,
        args.job_name,
        args.jobs_dir,
        agent=args.agent,
        model=args.model,
        agent_kwargs=args.agent_kwarg,
        allow_agent_hosts=args.allow_agent_host,
        install_only=args.install_only,
        n_attempts=args.n_attempts,
        n_concurrent=args.n_concurrent,
    )
    payload = run_harbor_preflight(
        command,
        args.jobs_dir / args.job_name,
        args.output,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["process_status"] == "exited" else 1


if __name__ == "__main__":
    raise SystemExit(main())
