from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mempool.adapters import OpenAICompatibleClient, OpenAICompatibleConfig
from mempool.benchmark import BenchmarkResult
from mempool.smoke_benchmark import (
    SmokeCodeTask,
    SmokeCodeBenchmarkAdapter,
    extract_python_code,
    task_to_dict,
)


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = """You write Python solutions for small benchmark tasks.
Return only Python code. Do not include explanations. Do not include markdown.
Define the requested function exactly. Do not read input or print output."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_env_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    loaded = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value
        loaded.append(key)
    return loaded


def evaluator_environment(
    required_packages: list[str] | None = None,
) -> dict[str, Any]:
    required_packages = required_packages or []
    packages = {}
    for package in required_packages:
        packages[package] = importlib.util.find_spec(package) is not None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "required_packages": packages,
    }


def missing_required_packages(required_packages: list[str]) -> list[str]:
    env = evaluator_environment(required_packages)
    return [
        package
        for package, available in env["required_packages"].items()
        if not available
    ]


def chat_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return ""


def record_key(worker_id: str, task_id: str, sample_index: int = 0) -> str:
    return f"{worker_id}::{task_id}::{sample_index}"


def request_failure_mode(error: Exception) -> str:
    if isinstance(error, TimeoutError) or "timed out" in str(error).lower():
        return "request_timeout"
    return "request_error"


def load_existing_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    existing = {}
    for worker in data.get("workers", []):
        for record in worker.get("records", []):
            existing[
                record_key(
                    record["worker_id"],
                    record["task"]["id"],
                    int(record.get("sample_index", 0)),
                )
            ] = record
    return existing


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outcomes(path: Path | None, summary: dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = flatten_outcomes(summary)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def summarize_worker(worker: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    solved = sum(1 for record in records if record["result"]["passed"])
    total_cost = sum(float(record["cost_usd"] or 0.0) for record in records)
    total_latency = sum(int(record["latency_ms"] or 0) for record in records)
    task_count = len(records)
    return {
        "worker_id": worker["id"],
        "model": worker["model"],
        "task_count": task_count,
        "solved": solved,
        "pass_at_1": solved / task_count if task_count else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_solved_task": round(total_cost / solved, 6) if solved else None,
        "mean_latency_ms": round(total_latency / task_count, 2) if task_count else None,
        "records": records,
    }


def run_task(
    client: OpenAICompatibleClient,
    worker: dict[str, Any],
    task: SmokeCodeTask,
    tasks_path: Path,
    eval_timeout_seconds: int,
    sample_index: int = 0,
) -> dict[str, Any]:
    prompt = (
        f"{task.prompt}\n\n"
        f"The evaluator will import and call `{task.function_name}` directly."
    )
    started = time.perf_counter()
    try:
        raw_response = client.chat(
            model=worker["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as error:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result = BenchmarkResult(
            task_id=task.id,
            passed=False,
            score=0.0,
            failure_mode=request_failure_mode(error),
            metadata={
                "error_type": type(error).__name__,
                "error": str(error)[-500:],
            },
        )
        return {
            "task": task_to_dict(task),
            "worker_id": worker["id"],
            "model": worker["model"],
            "sample_index": sample_index,
            "raw_output": "",
            "extracted_code": "",
            "result": asdict(result),
            "cost_usd": worker.get("cost_usd", 0.0),
            "latency_ms": elapsed_ms,
        }
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    raw_text = chat_content(raw_response)
    code = extract_python_code(raw_text)
    adapter = SmokeCodeBenchmarkAdapter(tasks_path, timeout_seconds=eval_timeout_seconds)
    result = adapter.evaluate_output(task, code)
    return {
        "task": task_to_dict(task),
        "worker_id": worker["id"],
        "model": worker["model"],
        "sample_index": sample_index,
        "raw_output": raw_text,
        "extracted_code": code,
        "result": asdict(result),
        "cost_usd": worker.get("cost_usd", 0.0),
        "latency_ms": elapsed_ms,
    }


def run_worker(
    client: OpenAICompatibleClient,
    worker: dict[str, Any],
    tasks_path: Path,
    limit: int | None,
    eval_timeout_seconds: int,
    repeat_count: int = 1,
    existing_records: dict[str, dict[str, Any]] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    adapter = SmokeCodeBenchmarkAdapter(tasks_path)
    tasks = adapter.load_tasks(limit=limit)
    records = []

    for task in tasks:
        for sample_index in range(repeat_count):
            key = record_key(worker["id"], task.id, sample_index)
            if existing_records and key in existing_records:
                if progress:
                    print(f"skip {worker['id']} {task.id} sample={sample_index}", flush=True)
                records.append(existing_records[key])
                continue
            if progress:
                print(f"run {worker['id']} {task.id} sample={sample_index}", flush=True)
            record = run_task(
                client,
                worker,
                task,
                tasks_path,
                eval_timeout_seconds,
                sample_index=sample_index,
            )
            if progress:
                status = "pass" if record["result"]["passed"] else "fail"
                print(
                    f"done {worker['id']} {task.id} sample={sample_index} "
                    f"{status} {record['latency_ms']}ms",
                    flush=True,
                )
            records.append(record)

    return summarize_worker(worker, records)


def rebuild_summary(
    config: dict[str, Any],
    tasks_path: Path,
    run_id: str,
    limit: int | None,
    repeat_count: int,
    existing_records: dict[str, dict[str, Any]],
    timestamp: str,
    evaluator_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapter = SmokeCodeBenchmarkAdapter(tasks_path)
    tasks = adapter.load_tasks(limit=limit)
    workers = []
    for worker in config["workers"]:
        records = []
        for task in tasks:
            for sample_index in range(repeat_count):
                existing = existing_records.get(record_key(worker["id"], task.id, sample_index))
                if existing:
                    records.append(existing)
        workers.append(summarize_worker(worker, records))
    return {
        "benchmark_id": "smoke-code-real-workers",
        "run_id": run_id,
        "timestamp": timestamp,
        "base_url": config["base_url"],
        "evaluator_env": evaluator_env or evaluator_environment(),
        "workers": workers,
    }


def flatten_outcomes(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for worker in summary["workers"]:
        for record in worker["records"]:
            result = record["result"]
            rows.append(
                {
                    "benchmark_id": summary["benchmark_id"],
                    "run_id": summary["run_id"],
                    "timestamp": summary["timestamp"],
                    "task_id": record["task"]["id"],
                    "task_family": record["task"]["family"],
                    "prompt": record["task"]["prompt"],
                    "worker_id": record["worker_id"],
                    "model": record["model"],
                    "sample_index": int(record.get("sample_index", 0)),
                    "workflow_kind": "route",
                    "passed": result["passed"],
                    "score": result["score"],
                    "failure_mode": result["failure_mode"],
                    "latency_ms": record["latency_ms"],
                    "cost_usd": record["cost_usd"],
                    "reward": result["score"],
                    "evaluator_python": summary.get("evaluator_env", {}).get("python_executable"),
                    "evaluator_python_version": summary.get("evaluator_env", {}).get("python_version"),
                    "evaluator_required_packages": summary.get("evaluator_env", {}).get("required_packages", {}),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run smoke benchmark on real workers.")
    parser.add_argument("--config", type=Path, default=ROOT / "research" / "evals" / "ollama_worker_pool.json")
    parser.add_argument("--tasks", type=Path, default=ROOT / "research" / "evals" / "smoke_code_tasks.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eval-timeout-seconds", type=int, default=5)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--run-id", default="ollama-smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--required-package", action="append", default=[])
    parser.add_argument("--allow-missing-required-package", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    load_env_file(args.env_file)
    missing_packages = missing_required_packages(args.required_package)
    if missing_packages and not args.allow_missing_required_package:
        raise SystemExit(
            "missing required evaluator package(s): "
            + ", ".join(missing_packages)
            + ". Use the benchmark environment or pass --allow-missing-required-package for an intentional negative control."
        )
    timestamp = utc_now()
    evaluator_env = evaluator_environment(args.required_package)
    existing_records = load_existing_records(args.output) if args.resume else {}
    summary = rebuild_summary(
        config=config,
        tasks_path=args.tasks,
        run_id=args.run_id,
        limit=args.limit,
        repeat_count=args.repeat_count,
        existing_records=existing_records,
        timestamp=timestamp,
        evaluator_env=evaluator_env,
    )
    client = OpenAICompatibleClient(
        OpenAICompatibleConfig(
            base_url=config["base_url"],
            api_key_env=config.get("api_key_env"),
            timeout_seconds=int(config.get("timeout_seconds", 180)),
            chat_options=config.get("chat_options"),
        )
    )

    workers = []
    for worker in config["workers"]:
        worker_summary = run_worker(
            client,
            worker,
            args.tasks,
            args.limit,
            args.eval_timeout_seconds,
            repeat_count=args.repeat_count,
            existing_records=existing_records,
            progress=args.progress,
        )
        workers.append(worker_summary)
        summary["workers"] = workers
        write_json(args.output, summary)
        write_outcomes(args.outcomes, summary)

    summary["workers"] = workers
    write_json(args.output, summary)
    write_outcomes(args.outcomes, summary)

    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
