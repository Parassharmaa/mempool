from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def seed_analysis(record: dict[str, Any]) -> dict[str, Any]:
    analysis = classify_task(
        {
            "id": record["task_id"],
            "prompt": record.get("prompt", ""),
            "tests": [],
        }
    )
    prompt_features = record.get("prompt_features") or {}
    if isinstance(prompt_features, dict):
        if prompt_features.get("categories"):
            analysis["categories"] = sorted(str(value) for value in prompt_features["categories"])
        if prompt_features.get("libraries"):
            analysis["libraries"] = sorted(str(value) for value in prompt_features["libraries"])
        if prompt_features.get("missing_libraries"):
            analysis["missing_libraries"] = sorted(
                str(value) for value in prompt_features["missing_libraries"]
            )
    analysis["seed_task_id"] = record["task_id"]
    analysis["top_worker_id"] = record.get("top_worker_id")
    analysis["second_worker_id"] = record.get("second_worker_id")
    analysis["best_ranked_alternate_worker_id"] = record.get("best_ranked_alternate_worker_id")
    analysis["fastest_passed_alternate_worker_id"] = record.get(
        "fastest_passed_alternate_worker_id"
    )
    return analysis


def corpus_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [record for record in records if bool(record.get("useful_any_fallback"))]
    second_positives = [record for record in records if bool(record.get("useful_second_fallback"))]
    hard_negatives = [record for record in records if bool(record.get("hard_negative"))]
    positive_seeds = [seed_analysis(record) for record in positives]
    second_positive_seeds = [seed_analysis(record) for record in second_positives]
    hard_negative_seeds = [seed_analysis(record) for record in hard_negatives]
    return {
        "task_ids": {str(record["task_id"]) for record in records},
        "positive_seeds": positive_seeds,
        "second_positive_seeds": second_positive_seeds,
        "hard_negative_seeds": hard_negative_seeds,
        "top_failure_counts": Counter(str(record.get("top_worker_id")) for record in records),
        "positive_top_counts": Counter(str(seed.get("top_worker_id")) for seed in positive_seeds),
        "positive_pair_counts": Counter(
            (
                str(seed.get("top_worker_id")),
                str(seed.get("best_ranked_alternate_worker_id")),
            )
            for seed in positive_seeds
            if seed.get("top_worker_id") and seed.get("best_ranked_alternate_worker_id")
        ),
        "second_pair_counts": Counter(
            (str(seed.get("top_worker_id")), str(seed.get("second_worker_id")))
            for seed in second_positive_seeds
            if seed.get("top_worker_id") and seed.get("second_worker_id")
        ),
        "hard_pair_counts": Counter(
            (str(seed.get("top_worker_id")), str(seed.get("second_worker_id")))
            for seed in hard_negative_seeds
            if seed.get("top_worker_id") and seed.get("second_worker_id")
        ),
    }


def max_similarity(seeds: list[dict[str, Any]], analysis: dict[str, Any]) -> float:
    return max((task_similarity(seed, analysis) for seed in seeds), default=0.0)


def seeds_for_top(seeds: list[dict[str, Any]], top_worker: str) -> list[dict[str, Any]]:
    return [seed for seed in seeds if str(seed.get("top_worker_id")) == top_worker]


def seeds_for_pair(
    seeds: list[dict[str, Any]],
    top_worker: str,
    alternate_worker: str,
    *,
    alternate_key: str,
) -> list[dict[str, Any]]:
    return [
        seed
        for seed in seeds
        if str(seed.get("top_worker_id")) == top_worker
        and str(seed.get(alternate_key)) == alternate_worker
    ]


def log_count_bonus(count: int, scale: float) -> float:
    if count <= 0:
        return 0.0
    return scale * min(4.0, count ** 0.5)


def score_candidate(
    *,
    analysis: dict[str, Any],
    routing: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    top_worker = str(routing["top_worker"])
    second_worker = str(routing["second_worker"])
    positive_same_top = seeds_for_top(profile["positive_seeds"], top_worker)
    positive_same_pair = seeds_for_pair(
        profile["positive_seeds"],
        top_worker,
        second_worker,
        alternate_key="best_ranked_alternate_worker_id",
    )
    second_same_pair = seeds_for_pair(
        profile["second_positive_seeds"],
        top_worker,
        second_worker,
        alternate_key="second_worker_id",
    )
    hard_same_top = seeds_for_top(profile["hard_negative_seeds"], top_worker)
    hard_same_pair = seeds_for_pair(
        profile["hard_negative_seeds"],
        top_worker,
        second_worker,
        alternate_key="second_worker_id",
    )
    positive_top_similarity = max_similarity(positive_same_top, analysis)
    positive_pair_similarity = max_similarity(positive_same_pair, analysis)
    second_pair_similarity = max_similarity(second_same_pair, analysis)
    hard_top_similarity = max_similarity(hard_same_top, analysis)
    hard_pair_similarity = max_similarity(hard_same_pair, analysis)
    global_positive_similarity = max_similarity(profile["positive_seeds"], analysis)
    global_hard_similarity = max_similarity(profile["hard_negative_seeds"], analysis)
    margin = float(routing["first_second_margin"])
    low_margin_bonus = 1.25 if margin <= 0.12 else 0.0
    uncertainty = max(0.0, 1.0 - margin) * 1.5
    pair = (top_worker, second_worker)
    pair_count_bonus = log_count_bonus(profile["positive_pair_counts"].get(pair, 0), 0.65)
    second_pair_count_bonus = log_count_bonus(profile["second_pair_counts"].get(pair, 0), 0.8)
    hard_pair_penalty = log_count_bonus(profile["hard_pair_counts"].get(pair, 0), 0.6)
    top_positive_count_bonus = log_count_bonus(profile["positive_top_counts"].get(top_worker, 0), 0.25)
    risk_penalty = float(analysis["environment_risk"]) * 0.45
    plausibility_penalty = float(analysis["plausibility_score"]) * 0.04
    missing_library_penalty = len(analysis.get("missing_libraries") or []) * 0.5
    score = (
        positive_pair_similarity * 3.0
        + second_pair_similarity * 2.0
        + positive_top_similarity * 1.25
        + global_positive_similarity * 0.35
        + uncertainty
        + low_margin_bonus
        + pair_count_bonus
        + second_pair_count_bonus
        + top_positive_count_bonus
        - hard_pair_similarity * 2.5
        - hard_top_similarity * 0.75
        - global_hard_similarity * 0.25
        - hard_pair_penalty
        - risk_penalty
        - plausibility_penalty
        - missing_library_penalty
    )
    return score, {
        "positive_top_similarity": round(positive_top_similarity, 4),
        "positive_pair_similarity": round(positive_pair_similarity, 4),
        "second_pair_similarity": round(second_pair_similarity, 4),
        "hard_top_similarity": round(hard_top_similarity, 4),
        "hard_pair_similarity": round(hard_pair_similarity, 4),
        "global_positive_similarity": round(global_positive_similarity, 4),
        "global_hard_similarity": round(global_hard_similarity, 4),
        "pair_count_bonus": round(pair_count_bonus, 4),
        "second_pair_count_bonus": round(second_pair_count_bonus, 4),
        "hard_pair_penalty": round(hard_pair_penalty, 4),
    }


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    analysis = item["analysis"]
    routing = item["routing"]
    return {
        "task_id": item["task"]["id"],
        "score": item["score"],
        "top_worker": routing["top_worker"],
        "top_probability": routing["top_probability"],
        "second_worker": routing["second_worker"],
        "second_probability": routing["second_probability"],
        "first_second_margin": routing["first_second_margin"],
        "categories": analysis["categories"],
        "libraries": analysis["libraries"],
        "missing_libraries": analysis["missing_libraries"],
        "environment_risk": analysis["environment_risk"],
        "plausibility_score": analysis["plausibility_score"],
        **{key: item[key] for key in item if key.endswith("_similarity")},
        "pair_count_bonus": item["pair_count_bonus"],
        "second_pair_count_bonus": item["second_pair_count_bonus"],
        "hard_pair_penalty": item["hard_pair_penalty"],
    }


def select_active_rescue_acquisition_batch(
    *,
    tasks: list[dict[str, Any]],
    router: Any,
    corpus_records: list[dict[str, Any]],
    exclude_ids: set[str],
    limit: int,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    profile = corpus_profile(corpus_records)
    candidates = []
    for task in tasks:
        task_id = str(task["id"])
        if task_id in exclude_ids or task_id in profile["task_ids"]:
            continue
        analysis = classify_task(task)
        routing = routing_summary(task, analysis, router)
        score, details = score_candidate(analysis=analysis, routing=routing, profile=profile)
        candidates.append(
            {
                "task": task,
                "analysis": analysis,
                "routing": routing,
                "score": round(score, 4),
                **details,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            -float(item["positive_pair_similarity"]),
            -float(item["second_pair_similarity"]),
            float(item["hard_pair_similarity"]),
            float(item["routing"]["first_second_margin"]),
            str(item["task"]["id"]),
        ),
    )
    selected = ranked[:limit]
    excluded_task_ids = set(exclude_ids) | set(profile["task_ids"])
    return {
        "candidate_count": len(candidates),
        "excluded_count": len(excluded_task_ids),
        "corpus_record_count": len(corpus_records),
        "corpus_unique_task_count": len(profile["task_ids"]),
        "positive_seed_count": len(profile["positive_seeds"]),
        "second_positive_seed_count": len(profile["second_positive_seeds"]),
        "hard_negative_seed_count": len(profile["hard_negative_seeds"]),
        "positive_top_counts": dict(sorted(profile["positive_top_counts"].items())),
        "positive_pair_counts": {
            f"{top}->{alternate}": count
            for (top, alternate), count in sorted(profile["positive_pair_counts"].items())
        },
        "second_pair_counts": {
            f"{top}->{second}": count
            for (top, second), count in sorted(profile["second_pair_counts"].items())
        },
        "hard_pair_counts": {
            f"{top}->{second}": count
            for (top, second), count in sorted(profile["hard_pair_counts"].items())
        },
        "selected_task_ids": [item["task"]["id"] for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [report_item(item) for item in selected],
        "ranked_candidates": [report_item(item) for item in ranked],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select fresh tasks likely to yield active-router top-fail alternate-pass rescues."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--tasks", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    task_paths = [path for group in args.tasks for path in group]
    router, active = load_active_router(args.registry)
    tasks = load_task_union(task_paths)
    corpus_records = read_jsonl(args.corpus)
    exclude_ids = read_routing_task_ids(args.exclude_routing_dataset)
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    selection = select_active_rescue_acquisition_batch(
        tasks=tasks,
        router=router,
        corpus_records=corpus_records,
        exclude_ids=exclude_ids,
        limit=args.limit,
    )
    report = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "task_sources": [str(path) for path in task_paths],
        "corpus": str(args.corpus),
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
