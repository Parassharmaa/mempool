from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.routing_dataset import read_routing_records


def read_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected task list in {path}")
    return data


def load_task_union(paths: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for task in read_tasks(path):
            by_id.setdefault(task["id"], task)
    return [by_id[task_id] for task_id in sorted(by_id)]


def excluded_task_ids(paths: list[Path]) -> set[str]:
    excluded = set()
    for path in paths:
        for record in read_routing_records(path):
            excluded.add(record["task_id"])
    return excluded


def read_outcome_task_ids(paths: list[Path]) -> set[str]:
    task_ids = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            task_ids.add(json.loads(line)["task_id"])
    return task_ids


def token_set(values: list[str]) -> set[str]:
    return {str(value).lower() for value in values}


def task_similarity(seed_analysis: dict[str, Any], candidate_analysis: dict[str, Any]) -> float:
    seed_libraries = token_set(seed_analysis["libraries"])
    candidate_libraries = token_set(candidate_analysis["libraries"])
    seed_categories = token_set(seed_analysis["categories"])
    candidate_categories = token_set(candidate_analysis["categories"])

    library_overlap = len(seed_libraries & candidate_libraries)
    category_overlap = len(seed_categories & candidate_categories)
    primary_match = seed_analysis["primary_category"] == candidate_analysis["primary_category"]
    risk_gap = abs(
        float(seed_analysis["environment_risk"])
        - float(candidate_analysis["environment_risk"])
    )
    plausibility_gap = abs(
        float(seed_analysis["plausibility_score"])
        - float(candidate_analysis["plausibility_score"])
    )
    return (
        library_overlap * 3.0
        + category_overlap * 1.5
        + (1.0 if primary_match else 0.0)
        - risk_gap * 0.5
        - plausibility_gap * 0.1
    )


def rank_similar_tasks(
    tasks: list[dict[str, Any]],
    seed_task_id: str,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude_ids = exclude_ids or set()
    by_id = {task["id"]: task for task in tasks}
    if seed_task_id not in by_id:
        raise ValueError(f"seed task {seed_task_id} not found")
    seed_analysis = classify_task(by_id[seed_task_id])

    ranked = []
    for task in tasks:
        if task["id"] == seed_task_id or task["id"] in exclude_ids:
            continue
        analysis = classify_task(task)
        ranked.append(
            {
                "task": task,
                "analysis": analysis,
                "score": round(task_similarity(seed_analysis, analysis), 4),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            -float(item["score"]),
            float(item["analysis"]["environment_risk"]),
            float(item["analysis"]["plausibility_score"]),
            item["task"]["id"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Select tasks similar to a seed task.")
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-task-id", required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_task_union(args.tasks)
    exclude_ids = excluded_task_ids(args.exclude_routing_dataset)
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    ranked = rank_similar_tasks(tasks, args.seed_task_id, exclude_ids=exclude_ids)
    selected = [item["task"] for item in ranked[: args.limit]]
    report = {
        "seed_task_id": args.seed_task_id,
        "task_sources": [str(path) for path in args.tasks],
        "excluded_routing_datasets": [str(path) for path in args.exclude_routing_dataset],
        "excluded_outcomes": [str(path) for path in args.exclude_outcomes],
        "candidate_count": len(ranked),
        "selected_task_ids": [task["id"] for task in selected],
        "ranked": [
            {
                "task_id": item["task"]["id"],
                "score": item["score"],
                "libraries": item["analysis"]["libraries"],
                "categories": item["analysis"]["categories"],
                "environment_risk": item["analysis"]["environment_risk"],
                "plausibility_score": item["analysis"]["plausibility_score"],
            }
            for item in ranked
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
