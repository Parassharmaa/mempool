from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.logits_router import LogitsRouter
from mempool.routing_dataset import read_routing_records

try:
    from tools.evaluate_active_policy import load_active_router
    from tools.select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from tools.select_similar_tasks import task_similarity
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router
    from select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from select_similar_tasks import task_similarity


def read_routing_task_ids(paths: list[Path]) -> set[str]:
    task_ids = set()
    for path in paths:
        for record in read_routing_records(path):
            task_ids.add(str(record["task_id"]))
    return task_ids


def task_record(task: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["id"],
        "benchmark_id": "bigcodebench-hard",
        "task_family": task.get("family", "bigcodebench_hard"),
        "prompt": task["prompt"],
        "prompt_features": {
            "categories": analysis["categories"],
            "libraries": analysis["libraries"],
            "missing_libraries": analysis["missing_libraries"],
            "environment_risk": analysis["environment_risk"],
            "plausibility_score": analysis["plausibility_score"],
        },
    }


def routing_summary(task: dict[str, Any], analysis: dict[str, Any], router: LogitsRouter) -> dict[str, Any]:
    distribution = router.distribution(task_record(task, analysis))
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    top_worker, top_probability = ranked[0]
    second_worker, second_probability = ranked[1] if len(ranked) > 1 else ("", 0.0)
    return {
        "distribution": distribution,
        "top_worker": top_worker,
        "top_probability": top_probability,
        "second_worker": second_worker,
        "second_probability": second_probability,
        "first_second_margin": top_probability - second_probability,
        "second_is_different_family": bool(second_worker and second_worker != top_worker),
    }


def seed_analyses(seed_datasets: list[Path], seed_task_ids: set[str] | None = None) -> list[dict[str, Any]]:
    seed_task_ids = seed_task_ids or set()
    seeds = []
    for path in seed_datasets:
        for record in read_routing_records(path):
            if seed_task_ids and record["task_id"] not in seed_task_ids:
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


def fallback_opportunity_score(
    analysis: dict[str, Any],
    routing: dict[str, Any],
    seeds: list[dict[str, Any]],
    preferred_second_workers: set[str],
) -> float:
    margin = float(routing["first_second_margin"])
    uncertainty = max(0.0, 1.0 - margin) * 8.0
    low_margin_bonus = 4.0 if margin <= 0.10 else 0.0
    second_worker_bonus = 2.0 if routing["second_worker"] in preferred_second_workers else 0.0
    seed_score = max((task_similarity(seed, analysis) for seed in seeds), default=0.0)
    risk_penalty = float(analysis["environment_risk"]) * 0.4
    return uncertainty + low_margin_bonus + second_worker_bonus + seed_score - risk_penalty


def select_fallback_opportunity_batch(
    tasks: list[dict[str, Any]],
    router: LogitsRouter,
    limit: int,
    exclude_ids: set[str] | None = None,
    seeds: list[dict[str, Any]] | None = None,
    preferred_second_workers: set[str] | None = None,
) -> dict[str, Any]:
    exclude_ids = exclude_ids or set()
    seeds = seeds or []
    preferred_second_workers = preferred_second_workers or set()
    candidates = []
    for task in tasks:
        if task["id"] in exclude_ids:
            continue
        analysis = classify_task(task)
        routing = routing_summary(task, analysis, router)
        score = fallback_opportunity_score(analysis, routing, seeds, preferred_second_workers)
        candidates.append(
            {
                "task": task,
                "analysis": analysis,
                "routing": routing,
                "score": round(score, 4),
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            float(item["routing"]["first_second_margin"]),
            float(item["analysis"]["environment_risk"]),
            item["task"]["id"],
        ),
    )
    selected = ranked[:limit]
    return {
        "candidate_count": len(candidates),
        "excluded_count": len(exclude_ids),
        "selected_task_ids": [item["task"]["id"] for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [
            {
                "task_id": item["task"]["id"],
                "score": item["score"],
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
            for item in selected
        ],
        "ranked_candidates": [
            {
                "task_id": item["task"]["id"],
                "score": item["score"],
                "top_worker": item["routing"]["top_worker"],
                "second_worker": item["routing"]["second_worker"],
                "first_second_margin": item["routing"]["first_second_margin"],
                "categories": item["analysis"]["categories"],
                "libraries": item["analysis"]["libraries"],
                "environment_risk": item["analysis"]["environment_risk"],
                "plausibility_score": item["analysis"]["plausibility_score"],
            }
            for item in ranked
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select fresh BigCodeBench tasks likely to create fallback-opportunity labels."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--seed-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--seed-task-id", action="append", default=[])
    parser.add_argument("--preferred-second-worker", action="append", default=[])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    router, active = load_active_router(args.registry)
    tasks = load_task_union(args.tasks)
    exclude_ids = read_routing_task_ids(args.exclude_routing_dataset)
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    seeds = seed_analyses(args.seed_routing_dataset, set(args.seed_task_id))
    preferred_second_workers = set(args.preferred_second_worker)
    selection = select_fallback_opportunity_batch(
        tasks,
        router,
        limit=args.limit,
        exclude_ids=exclude_ids,
        seeds=seeds,
        preferred_second_workers=preferred_second_workers,
    )
    report = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "task_sources": [str(path) for path in args.tasks],
        "excluded_routing_datasets": [str(path) for path in args.exclude_routing_dataset],
        "excluded_outcomes": [str(path) for path in args.exclude_outcomes],
        "seed_routing_datasets": [str(path) for path in args.seed_routing_dataset],
        "seed_task_ids": list(args.seed_task_id),
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
