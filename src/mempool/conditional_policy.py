from __future__ import annotations

from typing import Any

from .logits_router import LogitsRouter


def worker_by_id(record: dict[str, Any], worker_id: str) -> dict[str, Any]:
    for worker in record["workers"]:
        if worker["worker_id"] == worker_id:
            return worker
    raise KeyError(worker_id)


def ranked_workers(record: dict[str, Any], router: LogitsRouter) -> list[str]:
    distribution = router.distribution(record)
    return sorted(distribution, key=distribution.get, reverse=True)


def evaluate_conditional_fallback(
    records: list[dict[str, Any]],
    router: LogitsRouter,
    max_attempts: int = 2,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    solved = 0
    matched_target = 0
    solvable_task_count = 0
    solvable_solved = 0
    solvable_matched_target = 0
    total_latency = 0
    total_target_latency = 0
    total_latency_regret = 0
    attempts_used = 0
    examples = []

    for record in records:
        target_worker = worker_by_id(record, record["target_worker_id"])
        target_latency = int(target_worker["latency_ms"])
        solvable = any(bool(worker["passed"]) for worker in record["workers"])
        ranked = ranked_workers(record, router)
        attempts = []
        final_worker = None
        task_latency = 0
        for worker_id in ranked[:max_attempts]:
            worker = worker_by_id(record, worker_id)
            task_latency += int(worker["latency_ms"])
            attempts.append(
                {
                    "worker_id": worker_id,
                    "passed": bool(worker["passed"]),
                    "latency_ms": int(worker["latency_ms"]),
                }
            )
            final_worker = worker
            if worker["passed"]:
                break
        if final_worker is None:
            continue

        task_solved = bool(final_worker["passed"])
        final_worker_id = str(final_worker["worker_id"])
        solved += int(task_solved)
        matched_target += int(final_worker_id == record["target_worker_id"])
        if solvable:
            solvable_task_count += 1
            solvable_solved += int(task_solved)
            solvable_matched_target += int(final_worker_id == record["target_worker_id"])
        attempts_used += len(attempts)
        total_latency += task_latency
        total_target_latency += target_latency
        total_latency_regret += max(0, task_latency - target_latency)
        examples.append(
            {
                "task_id": record["task_id"],
                "target_worker_id": record["target_worker_id"],
                "attempts": attempts,
                "final_worker_id": final_worker_id,
                "solved": task_solved,
                "matched_target": final_worker_id == record["target_worker_id"],
            }
        )

    task_count = len(records)
    return {
        "policy": "conditional-fallback",
        "max_attempts": max_attempts,
        "task_count": task_count,
        "matched_target": matched_target,
        "target_accuracy": matched_target / task_count if task_count else 0.0,
        "solved": solved,
        "pass_at_1": solved / task_count if task_count else 0.0,
        "solvable_task_count": solvable_task_count,
        "solvable_solved": solvable_solved,
        "solvable_pass_at_1": solvable_solved / solvable_task_count if solvable_task_count else 0.0,
        "solvable_target_accuracy": solvable_matched_target / solvable_task_count if solvable_task_count else 0.0,
        "mean_attempts": attempts_used / task_count if task_count else 0.0,
        "mean_latency_ms": total_latency / task_count if task_count else 0.0,
        "mean_target_latency_ms": total_target_latency / task_count if task_count else 0.0,
        "mean_latency_regret_ms": total_latency_regret / task_count if task_count else 0.0,
        "examples": examples,
    }


def evaluate_gated_fallback(
    records: list[dict[str, Any]],
    router: LogitsRouter,
    max_attempts: int = 2,
    max_first_second_margin: float = 0.25,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    solved = 0
    matched_target = 0
    fallbacks_taken = 0
    fallback_opportunities = 0
    solvable_task_count = 0
    solvable_solved = 0
    solvable_matched_target = 0
    total_latency = 0
    total_target_latency = 0
    total_latency_regret = 0
    examples = []
    for record in records:
        distribution = router.distribution(record)
        ranked = sorted(distribution, key=distribution.get, reverse=True)
        if not ranked:
            continue
        attempts = []
        first = worker_by_id(record, ranked[0])
        task_latency = int(first.get("latency_ms", 0) or 0)
        attempts.append({"worker_id": ranked[0], "passed": bool(first["passed"])})
        final_worker = first
        if not first["passed"] and len(ranked) > 1 and max_attempts > 1:
            fallback_opportunities += 1
            margin = float(distribution[ranked[0]]) - float(distribution[ranked[1]])
            if margin <= max_first_second_margin:
                second = worker_by_id(record, ranked[1])
                attempts.append({"worker_id": ranked[1], "passed": bool(second["passed"])})
                final_worker = second
                task_latency += int(second.get("latency_ms", 0) or 0)
                fallbacks_taken += 1
        solved += int(bool(final_worker["passed"]))
        matched_target += int(final_worker["worker_id"] == record.get("target_worker_id"))
        target_worker = worker_by_id(record, record["target_worker_id"])
        target_latency = int(target_worker.get("latency_ms", 0) or 0)
        total_latency += task_latency
        total_target_latency += target_latency
        total_latency_regret += max(0, task_latency - target_latency)
        solvable = any(bool(worker.get("passed")) for worker in record.get("workers", []))
        if solvable:
            solvable_task_count += 1
            solvable_solved += int(bool(final_worker["passed"]))
            solvable_matched_target += int(final_worker["worker_id"] == record.get("target_worker_id"))
        examples.append(
            {
                "task_id": record["task_id"],
                "attempts": attempts,
                "final_worker_id": final_worker["worker_id"],
                "solved": bool(final_worker["passed"]),
            }
        )

    task_count = len(records)
    return {
        "policy": "gated-fallback",
        "task_count": task_count,
        "matched_target": matched_target,
        "target_accuracy": matched_target / task_count if task_count else 0.0,
        "solved": solved,
        "pass_at_1": solved / task_count if task_count else 0.0,
        "solvable_task_count": solvable_task_count,
        "solvable_solved": solvable_solved,
        "solvable_pass_at_1": solvable_solved / solvable_task_count if solvable_task_count else 0.0,
        "solvable_target_accuracy": solvable_matched_target / solvable_task_count if solvable_task_count else 0.0,
        "mean_latency_ms": total_latency / task_count if task_count else 0.0,
        "mean_target_latency_ms": total_target_latency / task_count if task_count else 0.0,
        "mean_latency_regret_ms": total_latency_regret / task_count if task_count else 0.0,
        "fallback_opportunities": fallback_opportunities,
        "fallbacks_taken": fallbacks_taken,
        "fallback_rate": fallbacks_taken / fallback_opportunities if fallback_opportunities else 0.0,
        "examples": examples,
    }
