from __future__ import annotations

from typing import Any


def target_pass_rate(record: dict[str, Any]) -> float:
    target = record.get("target_worker_id")
    for worker in record.get("workers", []):
        if worker.get("worker_id") == target:
            return float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0))
    return 0.0


def is_all_fail(record: dict[str, Any]) -> bool:
    return not any(
        float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0)) > 0.0
        for worker in record.get("workers", [])
    )


def filter_merge_ready_records(
    records: list[dict[str, Any]],
    *,
    min_target_pass_rate: float = 1.0,
    allow_all_fail_tasks: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = []
    dropped = []
    for record in records:
        reasons = []
        if is_all_fail(record) and not allow_all_fail_tasks:
            reasons.append("all_fail_task")
        rate = target_pass_rate(record)
        if rate < min_target_pass_rate:
            reasons.append("unstable_target")
        if reasons:
            dropped.append(
                {
                    "task_id": record.get("task_id"),
                    "target_worker_id": record.get("target_worker_id"),
                    "target_pass_rate": rate,
                    "reasons": reasons,
                }
            )
        else:
            kept.append(record)
    return kept, {
        "input_records": len(records),
        "kept_records": len(kept),
        "dropped_records": len(dropped),
        "dropped": dropped,
        "min_target_pass_rate": min_target_pass_rate,
        "allow_all_fail_tasks": allow_all_fail_tasks,
    }
