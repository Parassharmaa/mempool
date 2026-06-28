from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.outcome_mining import is_broad_pass_latency_row
from mempool.routing_dataset import read_routing_records

try:
    from tools.select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from tools.select_router_miss_neighborhood_batch import read_routing_task_ids
    from tools.select_similar_tasks import task_similarity
except ModuleNotFoundError:
    from select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from select_router_miss_neighborhood_batch import read_routing_task_ids
    from select_similar_tasks import task_similarity


def seed_analyses(dataset: Path, *, latency_safe: bool) -> list[dict[str, Any]]:
    seeds = []
    for record in read_routing_records(dataset):
        if is_broad_pass_latency_row(record) != latency_safe:
            continue
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


def score_candidate(task: dict[str, Any], seeds: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    analysis = classify_task(task)
    similarity = max((task_similarity(seed, analysis) for seed in seeds), default=0.0)
    score = similarity * 4.0 - float(analysis["environment_risk"]) * 0.5
    return score, analysis


def ranked_group(
    *,
    tasks: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    exclude_ids: set[str],
    label: str,
) -> list[dict[str, Any]]:
    ranked = []
    for task in tasks:
        task_id = str(task["id"])
        if task_id in exclude_ids:
            continue
        score, analysis = score_candidate(task, seeds)
        ranked.append(
            {
                "task": task,
                "label": label,
                "score": round(score, 4),
                "analysis": analysis,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            -float(item["score"]),
            float(item["analysis"]["environment_risk"]),
            str(item["task"]["id"]),
        ),
    )


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    analysis = item["analysis"]
    return {
        "task_id": str(item["task"]["id"]),
        "label": item["label"],
        "score": item["score"],
        "categories": analysis["categories"],
        "libraries": analysis["libraries"],
        "environment_risk": analysis["environment_risk"],
        "plausibility_score": analysis["plausibility_score"],
    }


def select_latency_safe_matched_controls(
    *,
    tasks: list[dict[str, Any]],
    seed_dataset: Path,
    exclude_ids: set[str],
    limit: int,
    per_label_limit: int,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if per_label_limit < 1:
        raise ValueError("per_label_limit must be at least 1")

    safe_seeds = seed_analyses(seed_dataset, latency_safe=True)
    unsafe_seeds = seed_analyses(seed_dataset, latency_safe=False)
    safe_ranked = ranked_group(
        tasks=tasks,
        seeds=safe_seeds,
        exclude_ids=exclude_ids,
        label="latency_safe_candidate",
    )
    unsafe_ranked = ranked_group(
        tasks=tasks,
        seeds=unsafe_seeds,
        exclude_ids=exclude_ids,
        label="unsafe_control_candidate",
    )

    selected = []
    selected_ids = set()
    groups = [safe_ranked, unsafe_ranked]
    while len(selected) < limit:
        added = False
        for group in groups:
            if len(selected) >= limit:
                break
            label_count = sum(1 for item in selected if item["label"] == group[0]["label"]) if group else 0
            if label_count >= per_label_limit:
                continue
            for item in group:
                task_id = str(item["task"]["id"])
                if task_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(task_id)
                added = True
                break
        if not added:
            break

    return {
        "seed_dataset": str(seed_dataset),
        "safe_seed_count": len(safe_seeds),
        "unsafe_seed_count": len(unsafe_seeds),
        "candidate_count": len({str(item["task"]["id"]) for item in safe_ranked + unsafe_ranked}),
        "excluded_count": len(exclude_ids),
        "selected_task_ids": [str(item["task"]["id"]) for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [report_item(item) for item in selected],
        "ranked_latency_safe_candidates": [report_item(item) for item in safe_ranked[: max(limit, 20)]],
        "ranked_unsafe_control_candidates": [report_item(item) for item in unsafe_ranked[: max(limit, 20)]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select matched candidates for learning the latency-safe calibration condition."
    )
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-dataset", type=Path, required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", nargs="+", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", nargs="+", default=[])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--per-label-limit", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_task_union(args.tasks)
    exclude_routing_datasets = [path for group in args.exclude_routing_dataset for path in group]
    exclude_outcomes = [path for group in args.exclude_outcomes for path in group]
    exclude_ids = read_routing_task_ids(exclude_routing_datasets)
    exclude_ids.update(read_outcome_task_ids(exclude_outcomes))
    selection = select_latency_safe_matched_controls(
        tasks=tasks,
        seed_dataset=args.seed_dataset,
        exclude_ids=exclude_ids,
        limit=args.limit,
        per_label_limit=args.per_label_limit,
    )
    report = {
        "task_sources": [str(path) for path in args.tasks],
        "excluded_routing_datasets": [str(path) for path in exclude_routing_datasets],
        "excluded_outcomes": [str(path) for path in exclude_outcomes],
        **{key: value for key, value in selection.items() if key != "selected_tasks"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection["selected_tasks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "selected_tasks"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
