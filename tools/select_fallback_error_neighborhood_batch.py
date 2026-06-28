from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.routing_dataset import read_routing_records

try:
    from tools.evaluate_active_policy import load_active_router
    from tools.select_fallback_opportunity_batch import read_routing_task_ids, routing_summary
    from tools.select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from tools.select_similar_tasks import task_similarity
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router
    from select_fallback_opportunity_batch import read_routing_task_ids, routing_summary
    from select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from select_similar_tasks import task_similarity


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mined_records_by_task(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = str(record["task_id"])
        previous = records.get(task_id)
        if previous is None or (
            bool(record.get("useful_any_fallback"))
            and not bool(previous.get("useful_any_fallback"))
        ):
            records[task_id] = record
    return records


def loo_error_task_ids(report: dict[str, Any]) -> dict[str, set[str]]:
    examples = report["leave_one_out"]["metrics"].get("examples", [])
    false_negatives = {
        str(example["task_id"])
        for example in examples
        if bool(example["actual_useful_fallback"]) and not bool(example["predicted_fallback"])
    }
    false_positives = {
        str(example["task_id"])
        for example in examples
        if (not bool(example["actual_useful_fallback"])) and bool(example["predicted_fallback"])
    }
    return {
        "false_negative": false_negatives,
        "false_positive": false_positives,
    }


def seed_analyses(
    mined_by_task: dict[str, dict[str, Any]],
    task_ids: set[str],
) -> list[dict[str, Any]]:
    analyses = []
    for task_id in sorted(task_ids):
        record = mined_by_task.get(task_id)
        if record is None:
            continue
        analysis = classify_task(
            {
                "id": record["task_id"],
                "prompt": record["prompt"],
                "tests": [],
            }
        )
        analysis["seed_task_id"] = task_id
        analysis["best_ranked_alternate_worker_id"] = record.get(
            "best_ranked_alternate_worker_id"
        )
        analysis["second_worker_id"] = record.get("second_worker_id")
        return_analysis = dict(analysis)
        analyses.append(return_analysis)
    return analyses


def max_similarity(seed_group: list[dict[str, Any]], analysis: dict[str, Any]) -> float:
    return max((task_similarity(seed, analysis) for seed in seed_group), default=0.0)


def rescue_worker_bonus(routing: dict[str, Any], false_negative_seeds: list[dict[str, Any]]) -> float:
    rescue_workers = {
        str(seed.get("best_ranked_alternate_worker_id"))
        for seed in false_negative_seeds
        if seed.get("best_ranked_alternate_worker_id")
    }
    second_worker = str(routing.get("second_worker", ""))
    return 2.0 if second_worker in rescue_workers else 0.0


def score_candidate(
    analysis: dict[str, Any],
    routing: dict[str, Any],
    false_negative_seeds: list[dict[str, Any]],
    false_positive_seeds: list[dict[str, Any]],
    target_group: str,
) -> float:
    margin = float(routing["first_second_margin"])
    uncertainty = max(0.0, 1.0 - margin) * 2.0
    low_margin_bonus = 1.0 if margin <= 0.12 else 0.0
    risk_penalty = float(analysis["environment_risk"]) * 0.5
    if target_group == "missed-positive-neighborhood":
        return (
            max_similarity(false_negative_seeds, analysis) * 3.0
            + rescue_worker_bonus(routing, false_negative_seeds)
            + uncertainty
            + low_margin_bonus
            - risk_penalty
        )
    if target_group == "false-positive-neighborhood":
        return (
            max_similarity(false_positive_seeds, analysis) * 3.0
            + uncertainty
            + low_margin_bonus
            - risk_penalty
        )
    raise ValueError(f"unknown target group: {target_group}")


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": item["task"]["id"],
        "target_group": item["target_group"],
        "score": item["score"],
        "false_negative_similarity": item["false_negative_similarity"],
        "false_positive_similarity": item["false_positive_similarity"],
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


def select_error_neighborhood_batch(
    tasks: list[dict[str, Any]],
    router: Any,
    false_negative_seeds: list[dict[str, Any]],
    false_positive_seeds: list[dict[str, Any]],
    limit: int,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    exclude_ids = exclude_ids or set()
    candidates = []
    for task in tasks:
        if task["id"] in exclude_ids:
            continue
        analysis = classify_task(task)
        routing = routing_summary(task, analysis, router)
        false_negative_similarity = max_similarity(false_negative_seeds, analysis)
        false_positive_similarity = max_similarity(false_positive_seeds, analysis)
        for target_group in ("missed-positive-neighborhood", "false-positive-neighborhood"):
            score = score_candidate(
                analysis,
                routing,
                false_negative_seeds,
                false_positive_seeds,
                target_group,
            )
            candidates.append(
                {
                    "task": task,
                    "analysis": analysis,
                    "routing": routing,
                    "target_group": target_group,
                    "false_negative_similarity": round(false_negative_similarity, 4),
                    "false_positive_similarity": round(false_positive_similarity, 4),
                    "score": round(score, 4),
                }
            )

    per_group_limit = max(1, limit // 2)
    selected = []
    selected_ids = set()
    for target_group in ("missed-positive-neighborhood", "false-positive-neighborhood"):
        group_ranked = sorted(
            [item for item in candidates if item["target_group"] == target_group],
            key=lambda item: (
                -float(item["score"]),
                float(item["routing"]["first_second_margin"]),
                float(item["analysis"]["environment_risk"]),
                item["task"]["id"],
            ),
        )
        for item in group_ranked:
            group_selected_count = sum(
                1 for selected_item in selected if selected_item["target_group"] == target_group
            )
            if group_selected_count >= per_group_limit:
                break
            if item["task"]["id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["task"]["id"])

    if len(selected) < limit:
        ranked_all = sorted(
            candidates,
            key=lambda item: (
                -float(item["score"]),
                float(item["routing"]["first_second_margin"]),
                float(item["analysis"]["environment_risk"]),
                item["task"]["id"],
            ),
        )
        for item in ranked_all:
            if len(selected) >= limit:
                break
            if item["task"]["id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["task"]["id"])

    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            item["target_group"],
            -float(item["score"]),
            float(item["routing"]["first_second_margin"]),
            float(item["analysis"]["environment_risk"]),
            item["task"]["id"],
        ),
    )
    return {
        "candidate_count": len({item["task"]["id"] for item in candidates}),
        "scored_candidate_count": len(candidates),
        "excluded_count": len(exclude_ids),
        "false_negative_seed_ids": [seed["seed_task_id"] for seed in false_negative_seeds],
        "false_positive_seed_ids": [seed["seed_task_id"] for seed in false_positive_seeds],
        "selected_task_ids": [item["task"]["id"] for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [report_item(item) for item in selected],
        "ranked_candidates": [report_item(item) for item in ranked_candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select fresh fallback-action candidates near mined-head LOO errors."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--mined-cases", type=Path, required=True)
    parser.add_argument("--mined-head-report", type=Path, required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    router, active = load_active_router(args.registry)
    tasks = load_task_union(args.tasks)
    mined_by_task = mined_records_by_task(args.mined_cases)
    error_ids = loo_error_task_ids(read_json(args.mined_head_report))
    false_negative_seeds = seed_analyses(mined_by_task, error_ids["false_negative"])
    false_positive_seeds = seed_analyses(mined_by_task, error_ids["false_positive"])
    exclude_ids = set(args.exclude_task_id)
    exclude_ids.update(read_routing_task_ids(args.exclude_routing_dataset))
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    exclude_ids.update(error_ids["false_negative"])
    exclude_ids.update(error_ids["false_positive"])

    selection = select_error_neighborhood_batch(
        tasks,
        router,
        false_negative_seeds,
        false_positive_seeds,
        limit=args.limit,
        exclude_ids=exclude_ids,
    )
    report = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "task_sources": [str(path) for path in args.tasks],
        "mined_cases": str(args.mined_cases),
        "mined_head_report": str(args.mined_head_report),
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
