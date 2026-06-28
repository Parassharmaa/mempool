from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.routing_dataset import read_routing_records


DEFAULT_CATEGORIES = ("filesystem", "datasci", "plotting", "subprocess", "network", "general")


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


def read_routing_task_ids(paths: list[Path]) -> set[str]:
    task_ids = set()
    for path in paths:
        for record in read_routing_records(path):
            task_ids.add(str(record["task_id"]))
    return task_ids


def read_outcome_task_ids(paths: list[Path]) -> set[str]:
    task_ids = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" in row:
                task_ids.add(str(row["task_id"]))
    return task_ids


def score_candidate(analysis: dict[str, Any], seen_libraries: set[str], seen_categories: set[str]) -> float:
    libraries = {str(item).lower() for item in analysis["libraries"]}
    categories = {str(item).lower() for item in analysis["categories"]}
    new_libraries = len(libraries - seen_libraries)
    new_categories = len(categories - seen_categories)
    return (
        new_categories * 5.0
        + new_libraries * 1.2
        - float(analysis["environment_risk"]) * 1.5
        - float(analysis["plausibility_score"]) * 0.15
    )


def candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, str]:
    analysis = item["analysis"]
    return (
        float(analysis["environment_risk"]),
        float(analysis["plausibility_score"]),
        item["task"]["id"],
    )


def hard_candidate_score(
    analysis: dict[str, Any],
    seen_libraries: set[str],
    seen_categories: set[str],
) -> float:
    libraries = {str(item).lower() for item in analysis["libraries"]}
    categories = {str(item).lower() for item in analysis["categories"]}
    new_libraries = len(libraries - seen_libraries)
    new_categories = len(categories - seen_categories)
    return (
        float(analysis["environment_risk"]) * 4.0
        + float(analysis["plausibility_score"]) * 1.0
        + new_categories * 2.0
        + new_libraries * 0.8
    )


def select_hard_batch(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_libraries: set[str] = set()
    seen_categories: set[str] = set()

    while len(selected) < limit:
        remaining = [item for item in candidates if item["task"]["id"] not in selected_ids]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda item: (
                hard_candidate_score(item["analysis"], seen_libraries, seen_categories),
                float(item["analysis"]["environment_risk"]),
                float(item["analysis"]["plausibility_score"]),
                item["task"]["id"],
            ),
        )
        selected.append({**best, "reason": "highest hard-task novelty score"})
        selected_ids.add(best["task"]["id"])
        seen_libraries.update(str(value).lower() for value in best["analysis"]["libraries"])
        seen_categories.update(str(value).lower() for value in best["analysis"]["categories"])

    return selected


def select_diverse_batch(
    candidates: list[dict[str, Any]],
    limit: int,
    preferred_categories: tuple[str, ...] = DEFAULT_CATEGORIES,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_libraries: set[str] = set()
    seen_categories: set[str] = set()

    def add(item: dict[str, Any], reason: str) -> None:
        selected.append({**item, "reason": reason})
        selected_ids.add(item["task"]["id"])
        seen_libraries.update(str(value).lower() for value in item["analysis"]["libraries"])
        seen_categories.update(str(value).lower() for value in item["analysis"]["categories"])

    for category in preferred_categories:
        if len(selected) >= limit:
            break
        matching = [
            item
            for item in candidates
            if item["task"]["id"] not in selected_ids
            and category in {str(value).lower() for value in item["analysis"]["categories"]}
        ]
        if not matching:
            continue
        add(min(matching, key=candidate_sort_key), f"lowest-risk fresh {category} task")

    while len(selected) < limit:
        remaining = [item for item in candidates if item["task"]["id"] not in selected_ids]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda item: (
                score_candidate(item["analysis"], seen_libraries, seen_categories),
                -float(item["analysis"]["environment_risk"]),
                -float(item["analysis"]["plausibility_score"]),
                item["task"]["id"],
            ),
        )
        add(best, "highest novelty score among remaining fresh tasks")

    return selected


def select_fresh_batch(
    tasks: list[dict[str, Any]],
    limit: int,
    exclude_ids: set[str] | None = None,
    preferred_categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    strategy: str = "diverse",
) -> dict[str, Any]:
    exclude_ids = exclude_ids or set()
    candidates = []
    for task in tasks:
        if task["id"] in exclude_ids:
            continue
        analysis = classify_task(task)
        candidates.append({"task": task, "analysis": analysis})

    if strategy == "diverse":
        selected = select_diverse_batch(candidates, limit, preferred_categories=preferred_categories)
    elif strategy == "hard":
        selected = select_hard_batch(candidates, limit)
    else:
        raise ValueError(f"unknown selection strategy: {strategy}")

    return {
        "candidate_count": len(candidates),
        "excluded_count": len(exclude_ids),
        "selected_task_ids": [item["task"]["id"] for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [
            {
                "task_id": item["task"]["id"],
                "reason": item["reason"],
                "libraries": item["analysis"]["libraries"],
                "categories": item["analysis"]["categories"],
                "environment_risk": item["analysis"]["environment_risk"],
                "plausibility_score": item["analysis"]["plausibility_score"],
            }
            for item in selected
        ],
        "ranked_candidates": [
            {
                "task_id": item["task"]["id"],
                "libraries": item["analysis"]["libraries"],
                "categories": item["analysis"]["categories"],
                "environment_risk": item["analysis"]["environment_risk"],
                "plausibility_score": item["analysis"]["plausibility_score"],
            }
            for item in sorted(candidates, key=candidate_sort_key)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a fresh diverse BigCodeBench acquisition batch.")
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--preferred-category", action="append", default=[])
    parser.add_argument("--strategy", choices=["diverse", "hard"], default="diverse")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_task_union(args.tasks)
    exclude_ids = read_routing_task_ids(args.exclude_routing_dataset)
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    preferred_categories = tuple(args.preferred_category or DEFAULT_CATEGORIES)
    selection = select_fresh_batch(
        tasks,
        limit=args.limit,
        exclude_ids=exclude_ids,
        preferred_categories=preferred_categories,
        strategy=args.strategy,
    )
    report = {
        "task_sources": [str(path) for path in args.tasks],
        "excluded_routing_datasets": [str(path) for path in args.exclude_routing_dataset],
        "excluded_outcomes": [str(path) for path in args.exclude_outcomes],
        "preferred_categories": list(preferred_categories),
        "strategy": args.strategy,
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
