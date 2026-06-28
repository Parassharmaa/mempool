from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_RECORD_FIELDS = {
    "task_id",
    "benchmark_id",
    "task_family",
    "prompt",
    "prompt_features",
    "workers",
    "target_worker_id",
    "target_distribution",
}

REQUIRED_WORKER_FIELDS = {
    "worker_id",
    "model",
    "passed",
    "score",
    "latency_ms",
    "cost_usd",
    "failure_mode",
    "reward",
    "target_probability",
}


def read_routing_records(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_routing_record(record: dict[str, Any]) -> list[str]:
    errors = []
    missing = REQUIRED_RECORD_FIELDS - set(record)
    if missing:
        errors.append(f"missing record fields: {sorted(missing)}")

    workers = record.get("workers", [])
    if not isinstance(workers, list) or not workers:
        errors.append("workers must be a non-empty list")
        return errors

    distribution = record.get("target_distribution", {})
    if not isinstance(distribution, dict):
        errors.append("target_distribution must be an object")
        distribution = {}

    worker_ids = {worker.get("worker_id") for worker in workers}
    if record.get("target_worker_id") not in worker_ids:
        errors.append("target_worker_id must refer to a worker")

    if set(distribution) != worker_ids:
        errors.append("target_distribution keys must match workers")

    probability_sum = sum(float(value) for value in distribution.values())
    if abs(probability_sum - 1.0) > 1e-4:
        errors.append(f"target probabilities must sum to 1.0, got {probability_sum}")

    for index, worker in enumerate(workers):
        missing_worker = REQUIRED_WORKER_FIELDS - set(worker)
        if missing_worker:
            errors.append(f"worker {index} missing fields: {sorted(missing_worker)}")
        probability = float(worker.get("target_probability", -1.0))
        if probability < 0.0 or probability > 1.0:
            errors.append(f"worker {index} target_probability out of range")

    return errors


def validate_routing_records(records: list[dict[str, Any]]) -> list[str]:
    errors = []
    seen_task_ids = set()
    for index, record in enumerate(records):
        task_id = record.get("task_id")
        if task_id in seen_task_ids:
            errors.append(f"record {index} duplicates task_id {task_id}")
        seen_task_ids.add(task_id)
        for error in validate_routing_record(record):
            errors.append(f"record {index}: {error}")
    return errors
