from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from mempool.coordinator import BaselineCoordinator
from mempool.smoke_benchmark import (
    SmokeCodeBenchmarkAdapter,
    fixture_output,
    task_to_dict,
)
from mempool.task import Task
from mempool.worker import Worker, WorkerPool


ROOT = Path(__file__).resolve().parents[1]


WORKERS = {
    "cheap-baseline": Worker(
        id="cheap-baseline",
        adapter="fixture",
        strengths=("code_easy", "code_text", "general"),
        expected_latency_ms=40,
        expected_cost_usd=0.001,
    ),
    "strong-fixture": Worker(
        id="strong-fixture",
        adapter="fixture",
        strengths=("code_data", "code_easy", "code_text", "general"),
        expected_latency_ms=120,
        expected_cost_usd=0.04,
    ),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def select_worker(mode: str, task: Task) -> Worker | None:
    if mode in WORKERS:
        return WORKERS[mode]
    if mode != "rule-router":
        raise ValueError(f"unknown mode: {mode}")
    pool = WorkerPool(workers=tuple(WORKERS.values()))
    plan = BaselineCoordinator().plan(task, pool)
    if not plan.worker_ids:
        return None
    return WORKERS[plan.worker_ids[0]]


def run(mode: str, limit: int | None) -> dict[str, object]:
    adapter = SmokeCodeBenchmarkAdapter(ROOT / "research" / "evals" / "smoke_code_tasks.json")
    tasks = adapter.load_tasks(limit=limit)
    records = []

    for benchmark_task in tasks:
        task = Task(
            id=benchmark_task.id,
            prompt=benchmark_task.prompt,
            family=benchmark_task.family,
        )
        worker = select_worker(mode, task)
        started = time.perf_counter()
        output = fixture_output(worker.id, benchmark_task.id) if worker else ""
        result = adapter.evaluate_output(benchmark_task, output)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        cost = worker.expected_cost_usd if worker else 0.0
        latency = (worker.expected_latency_ms if worker else 0) + elapsed_ms
        records.append(
            {
                "task": task_to_dict(benchmark_task),
                "worker_id": worker.id if worker else None,
                "output": output,
                "result": asdict(result),
                "cost_usd": cost,
                "latency_ms": latency,
            }
        )

    solved = sum(1 for record in records if record["result"]["passed"])
    total_cost = sum(float(record["cost_usd"] or 0.0) for record in records)
    total_latency = sum(int(record["latency_ms"] or 0) for record in records)
    task_count = len(records)
    return {
        "benchmark_id": adapter.id,
        "mode": mode,
        "timestamp": utc_now(),
        "task_count": task_count,
        "pass_at_1": solved / task_count if task_count else 0.0,
        "solved": solved,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_solved_task": round(total_cost / solved, 6) if solved else None,
        "mean_latency_ms": round(total_latency / task_count, 2) if task_count else None,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local smoke benchmark.")
    parser.add_argument(
        "--mode",
        choices=("cheap-baseline", "strong-fixture", "rule-router"),
        required=True,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.mode, args.limit)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
