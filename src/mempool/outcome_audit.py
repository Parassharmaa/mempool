from __future__ import annotations

from collections import defaultdict
from typing import Any


REQUIRED_OUTCOME_FIELDS = {
    "benchmark_id",
    "run_id",
    "timestamp",
    "task_id",
    "task_family",
    "prompt",
    "worker_id",
    "model",
    "workflow_kind",
    "passed",
    "score",
    "failure_mode",
    "latency_ms",
    "cost_usd",
    "reward",
}


def row_has_required_packages(row: dict[str, Any], required_packages: list[str]) -> bool:
    if not required_packages:
        return True
    packages = row.get("evaluator_required_packages", {})
    if not isinstance(packages, dict):
        return False
    return all(bool(packages.get(package)) for package in required_packages)


def audit_outcome_rows(
    rows: list[dict[str, Any]],
    required_evaluator_packages: list[str] | None = None,
    min_workers_per_task: int = 1,
    min_samples_per_worker_task: int = 1,
) -> dict[str, Any]:
    required_evaluator_packages = required_evaluator_packages or []
    missing_field_counts: dict[str, int] = defaultdict(int)
    package_mismatch_rows = 0
    task_workers: dict[str, set[str]] = defaultdict(set)
    samples_by_worker_task: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        for field in REQUIRED_OUTCOME_FIELDS - set(row):
            missing_field_counts[field] += 1
        task_id = str(row.get("task_id", ""))
        worker_id = str(row.get("worker_id", ""))
        if task_id and worker_id:
            task_workers[task_id].add(worker_id)
            samples_by_worker_task[(task_id, worker_id)] += 1
        if not row_has_required_packages(row, required_evaluator_packages):
            package_mismatch_rows += 1

    task_worker_counts = {
        task_id: len(workers)
        for task_id, workers in sorted(task_workers.items())
    }
    underspecified_tasks = [
        task_id
        for task_id, count in task_worker_counts.items()
        if count < min_workers_per_task
    ]
    low_sample_pairs = [
        {"task_id": task_id, "worker_id": worker_id, "samples": samples}
        for (task_id, worker_id), samples in sorted(samples_by_worker_task.items())
        if samples < min_samples_per_worker_task
    ]
    ready = (
        bool(rows)
        and not missing_field_counts
        and package_mismatch_rows == 0
        and not underspecified_tasks
        and not low_sample_pairs
    )

    return {
        "ready_for_conversion": ready,
        "row_count": len(rows),
        "task_count": len(task_workers),
        "worker_count": len({row.get("worker_id") for row in rows if row.get("worker_id")}),
        "required_evaluator_packages": required_evaluator_packages,
        "missing_field_counts": dict(sorted(missing_field_counts.items())),
        "package_mismatch_rows": package_mismatch_rows,
        "min_workers_per_task": min_workers_per_task,
        "underspecified_tasks": underspecified_tasks,
        "min_samples_per_worker_task": min_samples_per_worker_task,
        "low_sample_pairs": low_sample_pairs,
        "task_worker_counts": task_worker_counts,
    }
