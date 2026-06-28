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


DEFAULT_SPECIALIST_WORKERS = (
    "ollama-cloud-kimi-k2.7-code",
    "ollama-cloud-glm-5.2",
    "ollama-cloud-deepseek-v4-pro",
)


def seed_analyses(paths: list[Path], task_ids: set[str] | None = None) -> list[dict[str, Any]]:
    task_ids = task_ids or set()
    analyses = []
    for path in paths:
        for record in read_routing_records(path):
            if task_ids and str(record["task_id"]) not in task_ids:
                continue
            analyses.append(
                classify_task(
                    {
                        "id": record["task_id"],
                        "prompt": record["prompt"],
                        "tests": [],
                    }
                )
            )
    return analyses


def max_similarity(seeds: list[dict[str, Any]], analysis: dict[str, Any]) -> float:
    return max((task_similarity(seed, analysis) for seed in seeds), default=0.0)


def specialist_probability(distribution: dict[str, float], specialist_workers: set[str]) -> float:
    return max((float(distribution.get(worker_id, 0.0)) for worker_id in specialist_workers), default=0.0)


def specialist_rank(distribution: dict[str, float], specialist_workers: set[str]) -> int | None:
    ranked = sorted(distribution, key=distribution.get, reverse=True)
    for index, worker_id in enumerate(ranked, start=1):
        if worker_id in specialist_workers:
            return index
    return None


def score_candidate(
    *,
    analysis: dict[str, Any],
    routing: dict[str, Any],
    qwen_fail_seeds: list[dict[str, Any]],
    qwen_anchor_seeds: list[dict[str, Any]],
    specialist_workers: set[str],
    qwen_worker: str,
) -> tuple[float, dict[str, Any]]:
    qwen_fail_similarity = max_similarity(qwen_fail_seeds, analysis)
    qwen_anchor_similarity = max_similarity(qwen_anchor_seeds, analysis)
    top_worker = str(routing["top_worker"])
    second_worker = str(routing["second_worker"])
    margin = float(routing["first_second_margin"])
    specialist_prob = specialist_probability(routing["distribution"], specialist_workers)

    qwen_top_bonus = 2.5 if top_worker == qwen_worker else 0.0
    qwen_near_bonus = 1.0 if second_worker == qwen_worker else 0.0
    specialist_near_bonus = 2.0 if second_worker in specialist_workers else 0.0
    low_margin_bonus = max(0.0, 1.0 - margin) * 3.0
    risk_penalty = float(analysis["environment_risk"]) * 0.7
    plausibility_penalty = float(analysis["plausibility_score"]) * 0.08

    score = (
        qwen_fail_similarity * 4.0
        - qwen_anchor_similarity * 3.0
        + qwen_top_bonus
        + qwen_near_bonus
        + specialist_near_bonus
        + specialist_prob * 4.0
        + low_margin_bonus
        - risk_penalty
        - plausibility_penalty
    )
    return score, {
        "qwen_fail_similarity": round(qwen_fail_similarity, 4),
        "qwen_anchor_similarity": round(qwen_anchor_similarity, 4),
        "specialist_probability": round(specialist_prob, 6),
    }


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    routing = item["routing"]
    analysis = item["analysis"]
    return {
        "task_id": item["task"]["id"],
        "score": item["score"],
        "qwen_fail_similarity": item["qwen_fail_similarity"],
        "qwen_anchor_similarity": item["qwen_anchor_similarity"],
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


def select_qwen_fail_contrast_batch(
    *,
    tasks: list[dict[str, Any]],
    router: Any,
    exclude_ids: set[str],
    qwen_fail_seeds: list[dict[str, Any]],
    qwen_anchor_seeds: list[dict[str, Any]],
    specialist_workers: set[str],
    qwen_worker: str,
    limit: int,
    max_specialist_rank: int | None = None,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    candidates = []
    for task in tasks:
        if str(task["id"]) in exclude_ids:
            continue
        analysis = classify_task(task)
        routing = routing_summary(task, analysis, router)
        rank = specialist_rank(routing["distribution"], specialist_workers)
        if max_specialist_rank is not None and (
            rank is None or rank > max_specialist_rank
        ):
            continue
        score, details = score_candidate(
            analysis=analysis,
            routing=routing,
            qwen_fail_seeds=qwen_fail_seeds,
            qwen_anchor_seeds=qwen_anchor_seeds,
            specialist_workers=specialist_workers,
            qwen_worker=qwen_worker,
        )
        candidates.append(
            {
                "task": task,
                "analysis": analysis,
                "routing": routing,
                "specialist_rank": rank,
                "score": round(score, 4),
                **details,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            -float(item["qwen_fail_similarity"]),
            float(item["qwen_anchor_similarity"]),
            float(item["routing"]["first_second_margin"]),
            float(item["analysis"]["environment_risk"]),
            str(item["task"]["id"]),
        ),
    )
    selected = ranked[:limit]
    return {
        "candidate_count": len(candidates),
        "excluded_count": len(exclude_ids),
        "qwen_worker": qwen_worker,
        "specialist_workers": sorted(specialist_workers),
        "max_specialist_rank": max_specialist_rank,
        "selected_task_ids": [str(item["task"]["id"]) for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [report_item(item) for item in selected],
        "ranked_candidates": [report_item(item) for item in ranked],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select fresh tasks similar to Qwen-fail specialist seeds but unlike Qwen-fast anchors."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--qwen-fail-seed-dataset", type=Path, action="append", required=True)
    parser.add_argument("--qwen-fail-seed-task-id", action="append", default=[])
    parser.add_argument("--qwen-anchor-seed-dataset", type=Path, action="append", default=[])
    parser.add_argument("--qwen-anchor-seed-task-id", action="append", default=[])
    parser.add_argument("--specialist-worker", action="append", default=[])
    parser.add_argument("--qwen-worker", default="ollama-cloud-qwen3-coder-480b")
    parser.add_argument("--max-specialist-rank", type=int)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    router, active = load_active_router(args.registry)
    tasks = load_task_union(args.tasks)
    exclude_ids = read_routing_task_ids(args.exclude_routing_dataset)
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    qwen_fail_seeds = seed_analyses(
        args.qwen_fail_seed_dataset,
        set(args.qwen_fail_seed_task_id),
    )
    qwen_anchor_seeds = seed_analyses(
        args.qwen_anchor_seed_dataset,
        set(args.qwen_anchor_seed_task_id),
    )
    specialist_workers = set(args.specialist_worker or DEFAULT_SPECIALIST_WORKERS)
    selection = select_qwen_fail_contrast_batch(
        tasks=tasks,
        router=router,
        exclude_ids=exclude_ids,
        qwen_fail_seeds=qwen_fail_seeds,
        qwen_anchor_seeds=qwen_anchor_seeds,
        specialist_workers=specialist_workers,
        qwen_worker=args.qwen_worker,
        limit=args.limit,
        max_specialist_rank=args.max_specialist_rank,
    )
    report = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "task_sources": [str(path) for path in args.tasks],
        "excluded_routing_datasets": [str(path) for path in args.exclude_routing_dataset],
        "excluded_outcomes": [str(path) for path in args.exclude_outcomes],
        "qwen_fail_seed_datasets": [str(path) for path in args.qwen_fail_seed_dataset],
        "qwen_fail_seed_task_ids": list(args.qwen_fail_seed_task_id),
        "qwen_anchor_seed_datasets": [str(path) for path in args.qwen_anchor_seed_dataset],
        "qwen_anchor_seed_task_ids": list(args.qwen_anchor_seed_task_id),
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
