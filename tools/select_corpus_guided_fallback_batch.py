from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.logits_router import LogitsRouter

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def corpus_seed(record: dict[str, Any]) -> dict[str, Any]:
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
    return {
        **analysis,
        "seed_task_id": record["task_id"],
        "top_worker_id": record.get("top_worker_id"),
        "second_worker_id": record.get("second_worker_id"),
        "best_ranked_alternate_worker_id": record.get("best_ranked_alternate_worker_id"),
        "fastest_passed_alternate_worker_id": record.get("fastest_passed_alternate_worker_id"),
        "second_latency_ms": record.get("second_latency_ms"),
        "best_ranked_alternate_latency_ms": record.get("best_ranked_alternate_latency_ms"),
    }


def corpus_profile(corpus: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [record for record in corpus if bool(record.get("useful_any_fallback"))]
    second_positives = [record for record in corpus if bool(record.get("useful_second_fallback"))]
    hard_negatives = [record for record in corpus if bool(record.get("hard_negative"))]
    return {
        "task_ids": {str(record["task_id"]) for record in corpus},
        "positive_seeds": [corpus_seed(record) for record in positives],
        "second_positive_seeds": [corpus_seed(record) for record in second_positives],
        "hard_negative_seeds": [corpus_seed(record) for record in hard_negatives],
        "preferred_second_workers": {
            str(record["second_worker_id"]) for record in second_positives if record.get("second_worker_id")
        },
        "preferred_alternate_workers": {
            str(record["best_ranked_alternate_worker_id"])
            for record in positives
            if record.get("best_ranked_alternate_worker_id")
        },
        "positive_alternate_counts": Counter(
            str(record["best_ranked_alternate_worker_id"])
            for record in positives
            if record.get("best_ranked_alternate_worker_id")
        ),
        "hard_negative_top_counts": Counter(
            str(record["top_worker_id"]) for record in hard_negatives if record.get("top_worker_id")
        ),
    }


def max_similarity(seeds: list[dict[str, Any]], analysis: dict[str, Any]) -> float:
    return max((task_similarity(seed, analysis) for seed in seeds), default=0.0)


def score_candidate(
    task: dict[str, Any],
    analysis: dict[str, Any],
    routing: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    positive_similarity = max_similarity(profile["positive_seeds"], analysis)
    second_positive_similarity = max_similarity(profile["second_positive_seeds"], analysis)
    hard_negative_similarity = max_similarity(profile["hard_negative_seeds"], analysis)
    margin = float(routing["first_second_margin"])
    uncertainty = max(0.0, 1.0 - margin) * 2.0
    low_margin_bonus = 1.5 if margin <= 0.15 else 0.0
    preferred_second_bonus = (
        1.0 if routing["second_worker"] in profile["preferred_second_workers"] else 0.0
    )
    preferred_alternate_bonus = (
        0.6 if routing["second_worker"] in profile["preferred_alternate_workers"] else 0.0
    )
    hard_negative_penalty = hard_negative_similarity * 1.5
    risk_penalty = float(analysis["environment_risk"]) * 0.35
    plausibility_penalty = float(analysis["plausibility_score"]) * 0.03
    missing_library_penalty = len(analysis.get("missing_libraries") or []) * 0.4
    score = (
        positive_similarity * 2.5
        + second_positive_similarity
        + uncertainty
        + low_margin_bonus
        + preferred_second_bonus
        + preferred_alternate_bonus
        - hard_negative_penalty
        - risk_penalty
        - plausibility_penalty
        - missing_library_penalty
    )
    return score, {
        "positive_similarity": round(positive_similarity, 4),
        "second_positive_similarity": round(second_positive_similarity, 4),
        "hard_negative_similarity": round(hard_negative_similarity, 4),
        "low_margin_bonus": low_margin_bonus,
        "preferred_second_bonus": preferred_second_bonus,
        "preferred_alternate_bonus": preferred_alternate_bonus,
    }


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    analysis = item["analysis"]
    routing = item["routing"]
    return {
        "task_id": item["task"]["id"],
        "score": item["score"],
        "positive_similarity": item["positive_similarity"],
        "second_positive_similarity": item["second_positive_similarity"],
        "hard_negative_similarity": item["hard_negative_similarity"],
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
    }


def select_corpus_guided_fallback_batch(
    *,
    tasks: list[dict[str, Any]],
    router: LogitsRouter,
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
        score, details = score_candidate(task, analysis, routing, profile)
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
            float(item["routing"]["first_second_margin"]),
            float(item["analysis"]["environment_risk"]),
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
        "preferred_second_workers": sorted(profile["preferred_second_workers"]),
        "preferred_alternate_workers": sorted(profile["preferred_alternate_workers"]),
        "positive_alternate_counts": dict(sorted(profile["positive_alternate_counts"].items())),
        "hard_negative_top_counts": dict(sorted(profile["hard_negative_top_counts"].items())),
        "selected_task_ids": [item["task"]["id"] for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [report_item(item) for item in selected],
        "ranked_candidates": [report_item(item) for item in ranked],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select fresh tasks guided by a mined fallback-opportunity corpus."
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

    router, active = load_active_router(args.registry)
    task_paths = [path for group in args.tasks for path in group]
    tasks = load_task_union(task_paths)
    corpus_records = read_jsonl(args.corpus)
    exclude_ids = read_routing_task_ids(args.exclude_routing_dataset)
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    selection = select_corpus_guided_fallback_batch(
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
