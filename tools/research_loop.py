from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "research" / "runs"


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class EvaluationResult:
    tag: str
    timestamp: str
    score: float
    status: str
    checks: dict[str, bool]
    details: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_dir(tag: str) -> Path:
    return RUNS_DIR / tag


def events_path(tag: str) -> Path:
    return run_dir(tag) / "events.jsonl"


def results_path(tag: str) -> Path:
    return run_dir(tag) / "results.tsv"


def append_event(tag: str, event_type: str, payload: dict[str, Any]) -> None:
    path = events_path(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": utc_now(), "type": event_type, "payload": payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def run_command(command: list[str]) -> CommandResult:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else os.pathsep.join([src_path, existing_pythonpath])
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def init_run(args: argparse.Namespace) -> int:
    directory = run_dir(args.tag)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluations").mkdir(exist_ok=True)

    if not results_path(args.tag).exists():
        results_path(args.tag).write_text(
            "timestamp\tcommit\tstatus\tscore\tdescription\tevaluation_file\n",
            encoding="utf-8",
        )

    append_event(args.tag, "run_initialized", {"tag": args.tag})
    print(f"initialized {directory.relative_to(ROOT)}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    init_if_missing(args.tag)

    tests = run_command(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    )
    demo = run_command([sys.executable, "-m", "mempool.demo"])
    smoke = run_smoke_suite(args.tag)

    ledger = ROOT / "research" / "logs" / "demo.jsonl"
    ledger_exists = ledger.exists()
    ledger_parseable = False
    if ledger_exists:
        try:
            last_line = ledger.read_text(encoding="utf-8").strip().splitlines()[-1]
            parsed = json.loads(last_line)
            ledger_parseable = parsed.get("type") == "workflow_planned"
        except (IndexError, json.JSONDecodeError):
            ledger_parseable = False
        finally:
            ledger.unlink(missing_ok=True)

    checks = {
        "unit_tests": tests.returncode == 0,
        "demo_runs": demo.returncode == 0,
        "ledger_written": ledger_exists,
        "ledger_parseable": ledger_parseable,
        "smoke_signal": smoke["passed"],
    }
    score = sum(1.0 for passed in checks.values() if passed) / len(checks)
    status = "pass" if all(checks.values()) else "fail"

    result = EvaluationResult(
        tag=args.tag,
        timestamp=utc_now(),
        score=score,
        status=status,
        checks=checks,
        details={
            "tests": asdict(tests),
            "demo": asdict(demo),
            "smoke": smoke,
        },
    )

    eval_dir = run_dir(args.tag) / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_file = eval_dir / f"{result.timestamp.replace(':', '-')}.json"
    eval_file.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    append_event(
        args.tag,
        "evaluation_completed",
        {"status": status, "score": score, "evaluation_file": str(eval_file.relative_to(ROOT))},
    )

    print(json.dumps({"status": status, "score": score, "checks": checks}, indent=2))
    return 0 if status == "pass" else 1


def run_smoke_suite(tag: str) -> dict[str, Any]:
    modes = ("cheap-baseline", "strong-fixture", "rule-router")
    output_dir = run_dir(tag) / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for mode in modes:
        output_path = output_dir / f"{mode}.json"
        command = [
            sys.executable,
            "tools/run_smoke_benchmark.py",
            "--mode",
            mode,
            "--output",
            str(output_path.relative_to(ROOT)),
        ]
        result = run_command(command)
        if result.returncode != 0:
            summaries[mode] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            continue
        data = json.loads(output_path.read_text(encoding="utf-8"))
        summaries[mode] = {
            "pass_at_1": data["pass_at_1"],
            "cost_per_solved_task": data["cost_per_solved_task"],
            "mean_latency_ms": data["mean_latency_ms"],
            "solved": data["solved"],
            "task_count": data["task_count"],
            "output_file": str(output_path.relative_to(ROOT)),
        }

    cheap = summaries.get("cheap-baseline", {})
    strong = summaries.get("strong-fixture", {})
    router = summaries.get("rule-router", {})
    passed = (
        cheap.get("pass_at_1") == 0.5
        and strong.get("pass_at_1") == 1.0
        and router.get("pass_at_1") == 1.0
        and float(router.get("cost_per_solved_task", 999.0))
        < float(strong.get("cost_per_solved_task", 0.0))
    )

    return {
        "passed": passed,
        "summaries": summaries,
        "signal": (
            "rule-router matches strong pass@1 with lower cost per solved task"
            if passed
            else "smoke benchmark signal missing or regressed"
        ),
    }


def record(args: argparse.Namespace) -> int:
    init_if_missing(args.tag)
    commit = current_commit()
    latest = latest_evaluation(args.tag)
    score = ""
    eval_file = ""
    if latest:
        data = json.loads(latest.read_text(encoding="utf-8"))
        score = str(data.get("score", ""))
        eval_file = str(latest.relative_to(ROOT))

    with results_path(args.tag).open("a", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                [
                    utc_now(),
                    commit,
                    args.status,
                    score,
                    args.description,
                    eval_file,
                ]
            )
            + "\n"
        )

    append_event(
        args.tag,
        "result_recorded",
        {
            "commit": commit,
            "status": args.status,
            "score": score,
            "description": args.description,
            "evaluation_file": eval_file,
        },
    )
    print(f"recorded {args.status} for {args.tag}")
    return 0


def status(args: argparse.Namespace) -> int:
    directory = run_dir(args.tag)
    payload = {
        "tag": args.tag,
        "exists": directory.exists(),
        "results": str(results_path(args.tag).relative_to(ROOT)),
        "latest_evaluation": None,
    }
    latest = latest_evaluation(args.tag)
    if latest:
        payload["latest_evaluation"] = str(latest.relative_to(ROOT))
        payload["latest_score"] = json.loads(latest.read_text(encoding="utf-8")).get("score")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def init_if_missing(tag: str) -> None:
    if not results_path(tag).exists():
        namespace = argparse.Namespace(tag=tag)
        init_run(namespace)


def current_commit() -> str:
    result = run_command(["git", "rev-parse", "--short", "HEAD"])
    if result.returncode == 0:
        return result.stdout.strip()
    return "no-commit"


def latest_evaluation(tag: str) -> Path | None:
    eval_dir = run_dir(tag) / "evaluations"
    if not eval_dir.exists():
        return None
    files = sorted(eval_dir.glob("*.json"))
    return files[-1] if files else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded research loop helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a research run.")
    init_parser.add_argument("--tag", required=True)
    init_parser.set_defaults(func=init_run)

    eval_parser = subparsers.add_parser("evaluate", help="Run fixed local evaluation.")
    eval_parser.add_argument("--tag", required=True)
    eval_parser.set_defaults(func=evaluate)

    record_parser = subparsers.add_parser("record", help="Record latest result.")
    record_parser.add_argument("--tag", required=True)
    record_parser.add_argument("--status", choices=("keep", "discard", "crash"), required=True)
    record_parser.add_argument("--description", required=True)
    record_parser.set_defaults(func=record)

    status_parser = subparsers.add_parser("status", help="Show run status.")
    status_parser.add_argument("--tag", required=True)
    status_parser.set_defaults(func=status)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
