from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.routing_dataset import read_routing_records

try:
    from tools.evaluate_active_policy import load_active_router
    from tools.select_fallback_opportunity_batch import (
        read_routing_task_ids,
        routing_summary,
    )
    from tools.select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from tools.select_similar_tasks import task_similarity
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router
    from select_fallback_opportunity_batch import read_routing_task_ids, routing_summary
    from select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from select_similar_tasks import task_similarity


def positive_seed_analyses(paths: list[Path]) -> list[dict[str, Any]]:
    seeds = []
    seen = set()
    for path in paths:
        for record in read_routing_records(path):
            if record["task_id"] in seen:
                continue
            if not any(bool(worker["passed"]) for worker in record["workers"]):
                continue
            seen.add(record["task_id"])
            seeds.append(
                classify_task(
                    {
                        "id": record["task_id"],
                        "prompt": record["prompt"],
                        "tests": [],
                    }
                )
            )
    return seeds


def positive_neighborhood_score(
    analysis: dict[str, Any],
    routing: dict[str, Any],
    positive_seeds: list[dict[str, Any]],
    preferred_second_workers: set[str],
) -> float:
    similarity = max(
        (task_similarity(seed, analysis) for seed in positive_seeds),
        default=0.0,
    )
    margin = float(routing["first_second_margin"])
    uncertainty = max(0.0, 1.0 - margin) * 3.0
    low_margin_bonus = 1.5 if margin <= 0.15 else 0.0
    second_worker_bonus = 1.5 if routing["second_worker"] in preferred_second_workers else 0.0
    risk_penalty = float(analysis["environment_risk"]) * 0.8
    return similarity * 2.5 + uncertainty + low_margin_bonus + second_worker_bonus - risk_penalty


def select_positive_neighborhood_batch(
    tasks: list[dict[str, Any]],
    router: Any,
    positive_seeds: list[dict[str, Any]],
    limit: int,
    exclude_ids: set[str] | None = None,
    preferred_second_workers: set[str] | None = None,
) -> dict[str, Any]:
    exclude_ids = exclude_ids or set()
    preferred_second_workers = preferred_second_workers or set()
    candidates = []
    for task in tasks:
        if task["id"] in exclude_ids:
            continue
        analysis = classify_task(task)
        routing = routing_summary(task, analysis, router)
        similarity = max(
            (task_similarity(seed, analysis) for seed in positive_seeds),
            default=0.0,
        )
        score = positive_neighborhood_score(
            analysis,
            routing,
            positive_seeds,
            preferred_second_workers,
        )
        candidates.append(
            {
                "task": task,
                "analysis": analysis,
                "routing": routing,
                "positive_similarity": round(similarity, 4),
                "score": round(score, 4),
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            -float(item["positive_similarity"]),
            float(item["routing"]["first_second_margin"]),
            float(item["analysis"]["environment_risk"]),
            item["task"]["id"],
        ),
    )
    selected = ranked[:limit]
    return {
        "candidate_count": len(candidates),
        "excluded_count": len(exclude_ids),
        "positive_seed_count": len(positive_seeds),
        "selected_task_ids": [item["task"]["id"] for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [
            report_item(item)
            for item in selected
        ],
        "ranked_candidates": [report_item(item) for item in ranked],
    }


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": item["task"]["id"],
        "score": item["score"],
        "positive_similarity": item["positive_similarity"],
        "top_worker": item["routing"]["top_worker"],
        "top_probability": item["routing"]["top_probability"],
        "second_worker": item["routing"]["second_worker"],
        "second_probability": item["routing"]["second_probability"],
        "first_second_margin": item["routing"]["first_second_margin"],
        "categories": item["analysis"]["categories"],
        "libraries": item["analysis"]["libraries"],
        "environment_risk": item["analysis"]["environment_risk"],
        "plausibility_score": item["analysis"]["plausibility_score"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select fallback-screen candidates near known-positive tasks."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--positive-routing-dataset", type=Path, action="append", required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--preferred-second-worker", action="append", default=[])
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    router, active = load_active_router(args.registry)
    tasks = load_task_union(args.tasks)
    exclude_ids = set(args.exclude_task_id)
    exclude_ids.update(read_routing_task_ids(args.exclude_routing_dataset))
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    positive_seeds = positive_seed_analyses(args.positive_routing_dataset)
    preferred_second_workers = set(args.preferred_second_worker)
    selection = select_positive_neighborhood_batch(
        tasks,
        router,
        positive_seeds,
        limit=args.limit,
        exclude_ids=exclude_ids,
        preferred_second_workers=preferred_second_workers,
    )
    report = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "task_sources": [str(path) for path in args.tasks],
        "positive_routing_datasets": [str(path) for path in args.positive_routing_dataset],
        "excluded_routing_datasets": [str(path) for path in args.exclude_routing_dataset],
        "excluded_outcomes": [str(path) for path in args.exclude_outcomes],
        "excluded_task_ids": sorted(exclude_ids),
        "preferred_second_workers": sorted(preferred_second_workers),
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
