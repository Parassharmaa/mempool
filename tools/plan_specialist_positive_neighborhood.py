from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.routing_dataset import read_routing_records

try:
    from tools.plan_solvability_aware_specialist_acquisition import read_screened_task_ids
    from tools.select_fresh_bigcodebench_batch import load_task_union
    from tools.select_similar_tasks import task_similarity
except ModuleNotFoundError:
    from plan_solvability_aware_specialist_acquisition import read_screened_task_ids
    from select_fresh_bigcodebench_batch import load_task_union
    from select_similar_tasks import task_similarity


def worker_positive_seed_analyses(
    records: list[dict[str, Any]],
    worker_id: str,
) -> list[dict[str, Any]]:
    seeds = []
    seen = set()
    for record in records:
        if record["task_id"] in seen:
            continue
        worker = next(
            (item for item in record.get("workers", []) if item.get("worker_id") == worker_id),
            None,
        )
        if not worker:
            continue
        pass_rate = float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0))
        if pass_rate <= 0.0:
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


def score_positive_neighborhood_candidate(
    seed_analyses: list[dict[str, Any]],
    candidate_analysis: dict[str, Any],
) -> tuple[float, float]:
    positive_similarity = max(
        (task_similarity(seed, candidate_analysis) for seed in seed_analyses),
        default=0.0,
    )
    score = (
        positive_similarity * 3.0
        - float(candidate_analysis["environment_risk"]) * 1.0
        - float(candidate_analysis["plausibility_score"]) * 0.1
    )
    return score, positive_similarity


def select_specialist_positive_neighborhood(
    *,
    tasks: list[dict[str, Any]],
    routing_records: list[dict[str, Any]],
    target_workers: list[str],
    exclude_task_ids: set[str],
    per_worker_limit: int,
) -> dict[str, Any]:
    if per_worker_limit < 1:
        raise ValueError("per_worker_limit must be at least 1")
    selected_by_worker = {}
    selected_ids = set()
    routing_task_ids = {str(record["task_id"]) for record in routing_records}
    excludes = set(exclude_task_ids) | routing_task_ids

    for worker_id in target_workers:
        seeds = worker_positive_seed_analyses(routing_records, worker_id)
        ranked = []
        for task in tasks:
            task_id = str(task["id"])
            if task_id in excludes or task_id in selected_ids:
                continue
            analysis = classify_task(task)
            score, positive_similarity = score_positive_neighborhood_candidate(seeds, analysis)
            ranked.append(
                {
                    "task": task,
                    "analysis": analysis,
                    "score": round(score, 4),
                    "positive_similarity": round(positive_similarity, 4),
                }
            )
        ranked.sort(
            key=lambda item: (
                -float(item["positive_similarity"]),
                -float(item["score"]),
                float(item["analysis"]["environment_risk"]),
                float(item["analysis"]["plausibility_score"]),
                str(item["task"]["id"]),
            )
        )
        selected = ranked[:per_worker_limit]
        selected_ids.update(str(item["task"]["id"]) for item in selected)
        selected_by_worker[worker_id] = [
            {
                "task_id": item["task"]["id"],
                "score": item["score"],
                "positive_similarity": item["positive_similarity"],
                "categories": item["analysis"]["categories"],
                "libraries": item["analysis"]["libraries"],
                "environment_risk": item["analysis"]["environment_risk"],
                "plausibility_score": item["analysis"]["plausibility_score"],
            }
            for item in selected
        ]

    selected_task_ids = [
        item["task_id"]
        for worker_id in target_workers
        for item in selected_by_worker.get(worker_id, [])
    ]
    tasks_by_id = {str(task["id"]): task for task in tasks}
    return {
        "benchmark_id": "bigcodebench-hard-specialist-positive-neighborhood",
        "purpose": "Select fresh specialist candidates near tasks the target worker has already passed.",
        "target_workers": target_workers,
        "per_worker_limit": per_worker_limit,
        "selected_by_worker": selected_by_worker,
        "selected_task_ids": selected_task_ids,
        "selected_tasks": [tasks_by_id[task_id] for task_id in selected_task_ids],
        "excluded_task_count": len(excludes),
        "seed_counts_by_worker": {
            worker_id: len(worker_positive_seed_analyses(routing_records, worker_id))
            for worker_id in target_workers
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan specialist candidates near worker-positive routing rows."
    )
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--routing-dataset", type=Path, required=True)
    parser.add_argument("--target-worker", action="append", required=True)
    parser.add_argument("--exclude-screening-summary", type=Path, action="append", default=[])
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--per-worker-limit", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    exclude_task_ids = set(args.exclude_task_id)
    for path in args.exclude_screening_summary:
        exclude_task_ids.update(read_screened_task_ids(path))
    tasks = load_task_union(args.tasks)
    routing_records = read_routing_records(args.routing_dataset)
    selection = select_specialist_positive_neighborhood(
        tasks=tasks,
        routing_records=routing_records,
        target_workers=args.target_worker,
        exclude_task_ids=exclude_task_ids,
        per_worker_limit=args.per_worker_limit,
    )
    report = {key: value for key, value in selection.items() if key != "selected_tasks"}
    report["task_sources"] = [str(path) for path in args.tasks]
    report["routing_dataset"] = str(args.routing_dataset)
    report["exclude_screening_summaries"] = [str(path) for path in args.exclude_screening_summary]
    report["tasks_output"] = str(args.output)

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
