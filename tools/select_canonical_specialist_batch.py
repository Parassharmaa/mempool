from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task

try:
    from tools.evaluate_active_policy import load_active_router
    from tools.select_fallback_opportunity_batch import read_routing_task_ids, routing_summary
    from tools.select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router
    from select_fallback_opportunity_batch import read_routing_task_ids, routing_summary
    from select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids


DEFAULT_SPECIALIST_WORKERS = (
    "ollama-cloud-deepseek-v4-pro",
    "ollama-cloud-glm-5.2",
)


def specialist_rank(distribution: dict[str, float], specialist_workers: set[str]) -> int | None:
    ranked = sorted(distribution, key=distribution.get, reverse=True)
    for index, worker_id in enumerate(ranked, start=1):
        if worker_id in specialist_workers:
            return index
    return None


def specialist_probability(distribution: dict[str, float], specialist_workers: set[str]) -> float:
    return max((float(distribution.get(worker_id, 0.0)) for worker_id in specialist_workers), default=0.0)


def has_filesystem_or_archive_signal(analysis: dict[str, Any]) -> bool:
    categories = {str(value).lower() for value in analysis["categories"]}
    libraries = {str(value).lower() for value in analysis["libraries"]}
    return (
        "filesystem" in categories
        or bool(libraries & {"zipfile", "tarfile", "gzip", "shutil", "pathlib", "os", "glob"})
    )


def score_candidate(
    analysis: dict[str, Any],
    routing: dict[str, Any],
    specialist_workers: set[str],
) -> float:
    distribution = routing["distribution"]
    rank = specialist_rank(distribution, specialist_workers)
    specialist_prob = specialist_probability(distribution, specialist_workers)
    margin = float(routing["first_second_margin"])
    low_margin_bonus = 2.0 if margin <= 0.15 else 0.0
    filesystem_bonus = 4.0 if has_filesystem_or_archive_signal(analysis) else 0.0
    specialist_rank_bonus = {1: 1.0, 2: 3.0, 3: 1.5}.get(rank, 0.0)
    specialist_prob_bonus = specialist_prob * 6.0
    risk_penalty = float(analysis["environment_risk"]) * 1.0
    plausibility_penalty = float(analysis["plausibility_score"]) * 0.1
    return (
        filesystem_bonus
        + specialist_rank_bonus
        + specialist_prob_bonus
        + low_margin_bonus
        - risk_penalty
        - plausibility_penalty
    )


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    routing = item["routing"]
    analysis = item["analysis"]
    return {
        "task_id": item["task"]["id"],
        "score": item["score"],
        "top_worker": routing["top_worker"],
        "top_probability": routing["top_probability"],
        "second_worker": routing["second_worker"],
        "second_probability": routing["second_probability"],
        "first_second_margin": routing["first_second_margin"],
        "specialist_rank": item["specialist_rank"],
        "specialist_probability": item["specialist_probability"],
        "categories": analysis["categories"],
        "libraries": analysis["libraries"],
        "environment_risk": analysis["environment_risk"],
        "plausibility_score": analysis["plausibility_score"],
    }


def select_canonical_specialist_batch(
    tasks: list[dict[str, Any]],
    router: Any,
    limit: int,
    exclude_ids: set[str] | None = None,
    specialist_workers: set[str] | None = None,
    max_specialist_rank: int | None = None,
) -> dict[str, Any]:
    exclude_ids = exclude_ids or set()
    specialist_workers = specialist_workers or set(DEFAULT_SPECIALIST_WORKERS)
    candidates = []
    for task in tasks:
        if task["id"] in exclude_ids:
            continue
        analysis = classify_task(task)
        routing = routing_summary(task, analysis, router)
        distribution = routing["distribution"]
        rank = specialist_rank(distribution, specialist_workers)
        if max_specialist_rank is not None and (
            rank is None or rank > max_specialist_rank
        ):
            continue
        candidate = {
            "task": task,
            "analysis": analysis,
            "routing": routing,
            "specialist_rank": rank,
            "specialist_probability": round(
                specialist_probability(distribution, specialist_workers), 6
            ),
        }
        candidate["score"] = round(score_candidate(analysis, routing, specialist_workers), 4)
        candidates.append(candidate)

    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            item["specialist_rank"] if item["specialist_rank"] is not None else 99,
            float(item["analysis"]["environment_risk"]),
            float(item["routing"]["first_second_margin"]),
            item["task"]["id"],
        ),
    )
    selected = ranked[:limit]
    return {
        "candidate_count": len(candidates),
        "excluded_count": len(exclude_ids),
        "specialist_workers": sorted(specialist_workers),
        "max_specialist_rank": max_specialist_rank,
        "selected_task_ids": [item["task"]["id"] for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [report_item(item) for item in selected],
        "ranked_candidates": [report_item(item) for item in ranked],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select fresh canonical-pass BigCodeBench tasks with specialist-worker pressure."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--specialist-worker", action="append", default=[])
    parser.add_argument("--max-specialist-rank", type=int)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    router, active = load_active_router(args.registry)
    tasks = load_task_union(args.tasks)
    exclude_ids = set(args.exclude_task_id)
    exclude_ids.update(read_routing_task_ids(args.exclude_routing_dataset))
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    specialist_workers = set(args.specialist_worker or DEFAULT_SPECIALIST_WORKERS)
    selection = select_canonical_specialist_batch(
        tasks,
        router,
        limit=args.limit,
        exclude_ids=exclude_ids,
        specialist_workers=specialist_workers,
        max_specialist_rank=args.max_specialist_rank,
    )
    report = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "task_sources": [str(path) for path in args.tasks],
        "excluded_routing_datasets": [str(path) for path in args.exclude_routing_dataset],
        "excluded_outcomes": [str(path) for path in args.exclude_outcomes],
        "excluded_task_ids": sorted(exclude_ids),
        **{key: value for key, value in selection.items() if key != "selected_tasks"},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection["selected_tasks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
