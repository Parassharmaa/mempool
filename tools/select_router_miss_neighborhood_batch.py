from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.routing_dataset import read_routing_records

try:
    from tools.select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from tools.select_similar_tasks import task_similarity
except ModuleNotFoundError:
    from select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from select_similar_tasks import task_similarity


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_routing_task_ids(paths: list[Path]) -> set[str]:
    task_ids = set()
    for path in paths:
        for record in read_routing_records(path):
            task_ids.add(str(record["task_id"]))
    return task_ids


def seed_records(dataset: Path, task_ids: set[str]) -> dict[str, dict[str, Any]]:
    seeds = {}
    for record in read_routing_records(dataset):
        task_id = str(record["task_id"])
        if task_id in task_ids:
            seeds[task_id] = {
                "id": task_id,
                "prompt": record["prompt"],
                "tests": [],
            }
    return seeds


def score_against_neighborhood(
    *,
    seed_analysis: dict[str, Any],
    candidate_analysis: dict[str, Any],
    neighborhood: dict[str, Any],
) -> float:
    score = task_similarity(seed_analysis, candidate_analysis)
    library_keys = [
        str(item.get("libraries", ""))
        for item in neighborhood.get("top_library_keys", [])
    ]
    wanted_libraries = {
        value
        for key in library_keys
        for value in key.split("+")
        if value
    }
    candidate_libraries = {str(value).lower() for value in candidate_analysis["libraries"]}
    library_overlap = len(wanted_libraries & candidate_libraries)
    category_key = str(neighborhood.get("category_key", ""))
    wanted_categories = {value for value in category_key.split("+") if value}
    candidate_categories = {str(value).lower() for value in candidate_analysis["categories"]}
    category_overlap = len(wanted_categories & candidate_categories)
    return (
        score
        + library_overlap * 1.5
        + category_overlap * 1.0
        - float(candidate_analysis["environment_risk"]) * 0.25
    )


def rank_neighborhood_candidates(
    *,
    tasks: list[dict[str, Any]],
    neighborhood: dict[str, Any],
    seed_tasks: dict[str, dict[str, Any]],
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    seed_ids = [str(task_id) for task_id in neighborhood.get("task_ids", [])]
    seed_analyses = [
        classify_task(seed_tasks[task_id])
        for task_id in seed_ids
        if task_id in seed_tasks
    ]
    if not seed_analyses:
        return []

    ranked = []
    for task in tasks:
        task_id = str(task["id"])
        if task_id in exclude_ids:
            continue
        analysis = classify_task(task)
        score = max(
            score_against_neighborhood(
                seed_analysis=seed_analysis,
                candidate_analysis=analysis,
                neighborhood=neighborhood,
            )
            for seed_analysis in seed_analyses
        )
        ranked.append(
            {
                "task": task,
                "analysis": analysis,
                "score": round(score, 4),
                "neighborhood": neighborhood,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            -float(item["score"]),
            float(item["analysis"]["environment_risk"]),
            float(item["analysis"]["plausibility_score"]),
            str(item["task"]["id"]),
        ),
    )


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    analysis = item["analysis"]
    neighborhood = item["neighborhood"]
    return {
        "task_id": str(item["task"]["id"]),
        "score": item["score"],
        "source_seed_task_ids": list(neighborhood.get("task_ids", [])),
        "target_worker_id": neighborhood.get("target_worker_id"),
        "predicted_worker_id": neighborhood.get("predicted_worker_id"),
        "category_key": neighborhood.get("category_key"),
        "selection_reason": neighborhood.get("selection_reason", "fresh_neighborhood_candidate"),
        "libraries": analysis["libraries"],
        "categories": analysis["categories"],
        "environment_risk": analysis["environment_risk"],
        "plausibility_score": analysis["plausibility_score"],
    }


def select_router_miss_neighborhood_batch(
    *,
    tasks: list[dict[str, Any]],
    miss_plan: dict[str, Any],
    exclude_ids: set[str],
    limit: int,
    per_neighborhood_limit: int = 1,
    fallback_to_seed_tasks: bool = False,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if per_neighborhood_limit < 1:
        raise ValueError("per_neighborhood_limit must be at least 1")

    seed_task_ids = {
        str(task_id)
        for neighborhood in miss_plan.get("neighborhoods", [])
        for task_id in neighborhood.get("task_ids", [])
    }
    seeds = seed_records(Path(miss_plan["dataset"]), seed_task_ids)
    ranked_by_neighborhood = [
        rank_neighborhood_candidates(
            tasks=tasks,
            neighborhood=neighborhood,
            seed_tasks=seeds,
            exclude_ids=exclude_ids | seed_task_ids,
        )
        for neighborhood in miss_plan.get("neighborhoods", [])
    ]

    selected = []
    selected_ids: set[str] = set()
    per_neighborhood_counts = [0 for _ in ranked_by_neighborhood]
    while len(selected) < limit:
        added = False
        for index, ranked in enumerate(ranked_by_neighborhood):
            if len(selected) >= limit:
                break
            if per_neighborhood_counts[index] >= per_neighborhood_limit:
                continue
            for item in ranked:
                task_id = str(item["task"]["id"])
                if task_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(task_id)
                per_neighborhood_counts[index] += 1
                added = True
                break
        if not added:
            break

    fallback_used = False
    if not selected and fallback_to_seed_tasks:
        task_by_id = {str(task["id"]): task for task in tasks}
        for neighborhood in miss_plan.get("neighborhoods", []):
            if len(selected) >= limit:
                break
            for seed_task_id in neighborhood.get("task_ids", []):
                seed_task_id = str(seed_task_id)
                task = task_by_id.get(seed_task_id)
                if not task or seed_task_id in selected_ids:
                    continue
                analysis = classify_task(task)
                selected.append(
                    {
                        "task": task,
                        "analysis": analysis,
                        "score": 0.0,
                        "neighborhood": {
                            **neighborhood,
                            "selection_reason": "fallback_seed_repeat",
                        },
                    }
                )
                selected_ids.add(seed_task_id)
                fallback_used = True
                break

    ranked_candidates = [
        report_item(item)
        for ranked in ranked_by_neighborhood
        for item in ranked[: max(limit, per_neighborhood_limit)]
    ]
    return {
        "candidate_count": len({item["task"]["id"] for ranked in ranked_by_neighborhood for item in ranked}),
        "excluded_count": len(exclude_ids),
        "selected_task_ids": [str(item["task"]["id"]) for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [report_item(item) for item in selected],
        "fallback_to_seed_tasks": fallback_to_seed_tasks,
        "fallback_used": fallback_used,
        "ranked_candidates": ranked_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select fresh tasks similar to held-out router miss neighborhoods."
    )
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--miss-plan", type=Path, required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--per-neighborhood-limit", type=int, default=1)
    parser.add_argument("--fallback-to-seed-tasks", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_task_union(args.tasks)
    exclude_ids = read_routing_task_ids(args.exclude_routing_dataset)
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    miss_plan = read_json(args.miss_plan)
    selection = select_router_miss_neighborhood_batch(
        tasks=tasks,
        miss_plan=miss_plan,
        exclude_ids=exclude_ids,
        limit=args.limit,
        per_neighborhood_limit=args.per_neighborhood_limit,
        fallback_to_seed_tasks=args.fallback_to_seed_tasks,
    )
    report = {
        "miss_plan": str(args.miss_plan),
        "task_sources": [str(path) for path in args.tasks],
        "excluded_routing_datasets": [str(path) for path in args.exclude_routing_dataset],
        "excluded_outcomes": [str(path) for path in args.exclude_outcomes],
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
