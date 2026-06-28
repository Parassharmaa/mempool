from __future__ import annotations

from collections import Counter
from typing import Any

from .bigcodebench import classify_task


def task_by_id(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
    for task in tasks:
        if str(task.get("id")) == task_id:
            return task
    raise ValueError(f"task not found: {task_id}")


def select_unmeasured_workers(
    current_pool: dict[str, Any],
    measured_worker_ids: set[str],
) -> list[dict[str, Any]]:
    workers = current_pool.get("workers", [])
    if not isinstance(workers, list):
        raise ValueError("worker pool must contain a workers list")
    return [
        worker
        for worker in workers
        if str(worker.get("id")) not in measured_worker_ids
    ]


def build_catalog_candidate_acquisition_plan(
    current_pool: dict[str, Any],
    task_sources: list[dict[str, Any]],
    task_ids: list[str],
    measured_worker_ids: set[str],
    repeat_count: int,
) -> dict[str, Any]:
    selected_tasks = [task_by_id(task_sources, task_id) for task_id in task_ids]
    selected_workers = select_unmeasured_workers(current_pool, measured_worker_ids)
    worker_pool = {
        key: value
        for key, value in current_pool.items()
        if key not in {"workers", "selection_report"}
    }
    worker_pool["workers"] = selected_workers
    return {
        "benchmark_id": "bigcodebench-hard-catalog-candidate-acquisition",
        "purpose": "Evaluate newly available catalog workers on existing regression slices before adding them to router evidence.",
        "repeat_count": repeat_count,
        "task_count": len(selected_tasks),
        "worker_count": len(selected_workers),
        "call_count": len(selected_tasks) * len(selected_workers) * repeat_count,
        "selected_task_ids": task_ids,
        "selected_worker_ids": [str(worker["id"]) for worker in selected_workers],
        "selected_tasks": selected_tasks,
        "worker_pool": worker_pool,
    }


def equivalent_task_id(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        return value.replace("bigcodebench-hard-", "").replace("BigCodeBench/", "BigCodeBench-")

    return normalize(str(left)) == normalize(str(right))


def _task_seen(task: dict[str, Any], task_ids: set[str]) -> bool:
    return any(equivalent_task_id(str(task.get("id")), seen) for seen in task_ids)


def _seed_misses(candidate_report: dict[str, Any], target_workers: set[str]) -> list[dict[str, Any]]:
    predictions = candidate_report.get("leave_one_out", {}).get("predictions", [])
    return [
        item
        for item in predictions
        if str(item.get("target_worker_id")) in target_workers
        and item.get("predicted_worker_id") != item.get("target_worker_id")
    ]


def build_specialist_acquisition_plan(
    task_sources: list[dict[str, Any]],
    routing_records: list[dict[str, Any]],
    candidate_report: dict[str, Any],
    target_workers: list[str],
    comparison_workers: list[str],
    exclude_task_ids: set[str],
    per_worker_limit: int,
    repeat_count: int = 1,
) -> dict[str, Any]:
    target_set = set(target_workers)
    seed_misses = _seed_misses(candidate_report, target_set)
    seen_ids = {str(record.get("task_id")) for record in routing_records} | set(exclude_task_ids)
    selected_by_worker: dict[str, list[dict[str, Any]]] = {worker: [] for worker in target_workers}
    selected_ids: list[str] = []
    for worker in target_workers:
        for task in task_sources:
            if len(selected_by_worker[worker]) >= per_worker_limit:
                break
            if _task_seen(task, seen_ids) or str(task.get("id")) in selected_ids:
                continue
            selected_by_worker[worker].append({"task_id": str(task["id"]), "task": task})
            selected_ids.append(str(task["id"]))
    worker_ids = list(dict.fromkeys([*target_workers, *comparison_workers]))
    return {
        "seed_miss_count": len(seed_misses),
        "selected_task_ids": selected_ids,
        "selected_by_worker": selected_by_worker,
        "worker_ids_to_run": worker_ids,
        "repeat_count": repeat_count,
        "call_count": len(selected_ids) * len(worker_ids) * repeat_count,
    }


def solvability_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    library_counts: Counter[str] = Counter()
    positive_count = 0
    for record in records:
        positive = any(bool(worker.get("passed")) or float(worker.get("pass_rate", 0.0) or 0.0) > 0 for worker in record.get("workers", []))
        if not positive:
            continue
        positive_count += 1
        features = record.get("prompt_features") or {}
        category_counts.update(str(value) for value in features.get("categories", []))
        library_counts.update(str(value) for value in features.get("libraries", []))
    return {
        "positive_count": positive_count,
        "category_counts": dict(category_counts),
        "library_counts": dict(library_counts),
    }


def _solvability_score(task: dict[str, Any], profile: dict[str, Any]) -> float:
    analysis = classify_task(task)
    score = 0.0
    for category in analysis["categories"]:
        score += float(profile["category_counts"].get(category, 0))
    for library in analysis["libraries"]:
        score += float(profile["library_counts"].get(library, 0)) * 2.0
    score -= float(analysis["environment_risk"]) * 0.5
    score -= float(analysis["plausibility_score"]) * 0.05
    return round(score, 4)


def build_solvability_aware_specialist_plan(
    task_sources: list[dict[str, Any]],
    routing_records: list[dict[str, Any]],
    candidate_report: dict[str, Any],
    target_workers: list[str],
    exclude_task_ids: set[str],
    per_worker_limit: int,
) -> dict[str, Any]:
    profile = solvability_profile(routing_records)
    seed_misses = _seed_misses(candidate_report, set(target_workers))
    seen_ids = {str(record.get("task_id")) for record in routing_records} | set(exclude_task_ids)
    selected_by_worker: dict[str, list[dict[str, Any]]] = {}
    selected_ids: list[str] = []
    scored = []
    for task in task_sources:
        if _task_seen(task, seen_ids):
            continue
        score = _solvability_score(task, profile)
        scored.append((score, str(task["id"]), task))
    scored.sort(key=lambda item: (-item[0], item[1]))
    for worker in target_workers:
        rows = []
        for score, task_id, task in scored:
            if len(rows) >= per_worker_limit:
                break
            if task_id in selected_ids:
                continue
            rows.append({"task_id": task_id, "task": task, "solvability_score": score})
            selected_ids.append(task_id)
        selected_by_worker[worker] = rows
    return {
        "seed_miss_count": len(seed_misses),
        "positive_prior_count": profile["positive_count"],
        "selected_task_ids": selected_ids,
        "selected_by_worker": selected_by_worker,
    }
