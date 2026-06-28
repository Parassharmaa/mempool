from __future__ import annotations

from typing import Any


def _worker_pass_rate(worker: dict[str, Any]) -> float:
    if "pass_rate" in worker:
        return float(worker["pass_rate"])
    return 1.0 if bool(worker.get("passed")) else 0.0


def audit_routing_merge_readiness(
    records: list[dict[str, Any]],
    min_target_pass_rate: float = 1.0,
    allow_all_fail_tasks: bool = False,
) -> dict[str, Any]:
    all_fail_tasks = []
    unstable_target_tasks = []
    target_failure_tasks = []
    solvable_tasks = 0

    for record in records:
        workers = record.get("workers", [])
        target_worker_id = str(record.get("target_worker_id", ""))
        target_worker = next(
            (worker for worker in workers if str(worker.get("worker_id")) == target_worker_id),
            None,
        )
        pass_rates = [_worker_pass_rate(worker) for worker in workers]
        if not any(rate > 0.0 for rate in pass_rates):
            all_fail_tasks.append(str(record.get("task_id")))
            continue
        solvable_tasks += 1
        if target_worker is None:
            target_failure_tasks.append(str(record.get("task_id")))
            continue
        target_pass_rate = _worker_pass_rate(target_worker)
        if target_pass_rate <= 0.0:
            target_failure_tasks.append(str(record.get("task_id")))
        elif target_pass_rate < min_target_pass_rate:
            unstable_target_tasks.append(
                {
                    "task_id": str(record.get("task_id")),
                    "target_worker_id": target_worker_id,
                    "target_pass_rate": target_pass_rate,
                }
            )

    reasons = []
    if all_fail_tasks and not allow_all_fail_tasks:
        reasons.append("dataset contains all-fail fastest-failure tasks")
    if target_failure_tasks:
        reasons.append("dataset contains solvable tasks whose target worker failed")
    if unstable_target_tasks:
        reasons.append("dataset contains unstable target workers")

    return {
        "ready_to_merge": not reasons,
        "record_count": len(records),
        "solvable_task_count": solvable_tasks,
        "min_target_pass_rate": min_target_pass_rate,
        "allow_all_fail_tasks": allow_all_fail_tasks,
        "all_fail_tasks": all_fail_tasks,
        "unstable_target_tasks": unstable_target_tasks,
        "target_failure_tasks": target_failure_tasks,
        "reasons": reasons,
    }
