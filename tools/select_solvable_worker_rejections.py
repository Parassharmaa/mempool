from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task

try:
    from tools.select_similar_tasks import task_similarity
    from tools.select_worker_rejection_tasks import read_json, read_jsonl, rejected_task_ids
except ModuleNotFoundError:
    from select_similar_tasks import task_similarity
    from select_worker_rejection_tasks import read_json, read_jsonl, rejected_task_ids


DEFAULT_SPECIALIST_WORKERS = {
    "ollama-cloud-kimi-k2.7-code",
    "ollama-cloud-glm-5.2",
    "ollama-cloud-deepseek-v4-pro",
}


def task_analysis_from_outcome(row: dict[str, Any]) -> dict[str, Any]:
    return classify_task(
        {
            "id": row["task_id"],
            "prompt": row.get("prompt", ""),
            "tests": [],
        }
    )


def seed_analyses_from_outcomes(
    rows: list[dict[str, Any]],
    *,
    specialist_workers: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("worker_id")) not in specialist_workers:
            continue
        by_task.setdefault(str(row["task_id"]), []).append(row)

    positive = []
    universal_fail = []
    for task_rows in by_task.values():
        if not task_rows:
            continue
        if any(bool(row.get("passed")) for row in task_rows):
            first_pass = next(row for row in task_rows if bool(row.get("passed")))
            positive.append(task_analysis_from_outcome(first_pass))
        elif all(not bool(row.get("passed")) for row in task_rows):
            universal_fail.append(task_analysis_from_outcome(task_rows[0]))
    return positive, universal_fail


def max_similarity(seeds: list[dict[str, Any]], analysis: dict[str, Any]) -> float:
    return max(0.0, *(task_similarity(seed, analysis) for seed in seeds))


def solvability_gate_score(
    analysis: dict[str, Any],
    *,
    positive_seeds: list[dict[str, Any]],
    universal_fail_seeds: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    positive_similarity = max_similarity(positive_seeds, analysis)
    universal_fail_similarity = max_similarity(universal_fail_seeds, analysis)
    score = (
        positive_similarity * 2.0
        - universal_fail_similarity * 2.5
        - float(analysis["environment_risk"]) * 0.6
        - float(analysis["plausibility_score"]) * 0.08
    )
    return score, {
        "positive_similarity": round(positive_similarity, 4),
        "universal_fail_similarity": round(universal_fail_similarity, 4),
    }


def select_solvable_worker_rejections(
    *,
    tasks: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    rejected_worker_id: str,
    specialist_workers: set[str],
    limit: int,
    min_gate_score: float | None = None,
    max_universal_fail_similarity: float | None = None,
    max_pass_latency_ms: float | None = None,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")

    rejected_ids = set(
        rejected_task_ids(
            qwen_rows,
            worker_id=rejected_worker_id,
            max_pass_latency_ms=max_pass_latency_ms,
        )
    )
    positive_seeds, universal_fail_seeds = seed_analyses_from_outcomes(
        prior_rows,
        specialist_workers=specialist_workers,
    )

    scored_rejections = []
    rejected_by_score = 0
    rejected_by_universal_fail_cap = 0
    candidates = []
    for task in tasks:
        task_id = str(task["id"])
        if task_id not in rejected_ids:
            continue
        analysis = classify_task(task)
        score, details = solvability_gate_score(
            analysis,
            positive_seeds=positive_seeds,
            universal_fail_seeds=universal_fail_seeds,
        )
        scored_item = {
            "task": task,
            "task_id": task_id,
            "score": round(score, 4),
            "categories": analysis["categories"],
            "libraries": analysis["libraries"],
            "environment_risk": analysis["environment_risk"],
            "plausibility_score": analysis["plausibility_score"],
            **details,
        }
        scored_rejections.append(scored_item)
        if min_gate_score is not None and score < min_gate_score:
            rejected_by_score += 1
            continue
        if (
            max_universal_fail_similarity is not None
            and details["universal_fail_similarity"] > max_universal_fail_similarity
        ):
            rejected_by_universal_fail_cap += 1
            continue
        candidates.append(scored_item)

    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            -float(item["positive_similarity"]),
            float(item["universal_fail_similarity"]),
            float(item["environment_risk"]),
            str(item["task_id"]),
        ),
    )
    selected = ranked[:limit]
    return {
        "rejected_worker_id": rejected_worker_id,
        "specialist_workers": sorted(specialist_workers),
        "max_pass_latency_ms": max_pass_latency_ms,
        "min_gate_score": min_gate_score,
        "max_universal_fail_similarity": max_universal_fail_similarity,
        "positive_seed_count": len(positive_seeds),
        "universal_fail_seed_count": len(universal_fail_seeds),
        "rejected_task_count": len(rejected_ids),
        "candidate_count": len(candidates),
        "rejected_by_score": rejected_by_score,
        "rejected_by_universal_fail_cap": rejected_by_universal_fail_cap,
        "selected_task_ids": [item["task_id"] for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "scored_rejections": [
            {key: value for key, value in item.items() if key != "task"}
            for item in sorted(
                scored_rejections,
                key=lambda item: (
                    -float(item["score"]),
                    -float(item["positive_similarity"]),
                    float(item["universal_fail_similarity"]),
                    float(item["environment_risk"]),
                    str(item["task_id"]),
                ),
            )
        ],
        "ranked_candidates": [{key: value for key, value in item.items() if key != "task"} for item in ranked],
        "selected": [{key: value for key, value in item.items() if key != "task"} for item in selected],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select rejected tasks that also resemble specialist-solvable prior outcomes."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--qwen-outcomes", type=Path, required=True)
    parser.add_argument("--prior-outcomes", type=Path, action="append", required=True)
    parser.add_argument("--rejected-worker-id", default="ollama-cloud-qwen3-coder-480b")
    parser.add_argument("--specialist-worker", action="append", default=[])
    parser.add_argument("--max-pass-latency-ms", type=float)
    parser.add_argument("--min-gate-score", type=float)
    parser.add_argument("--max-universal-fail-similarity", type=float)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    prior_rows = []
    for path in args.prior_outcomes:
        prior_rows.extend(read_jsonl(path))
    specialist_workers = set(args.specialist_worker or DEFAULT_SPECIALIST_WORKERS)
    selection = select_solvable_worker_rejections(
        tasks=read_json(args.tasks),
        qwen_rows=read_jsonl(args.qwen_outcomes),
        prior_rows=prior_rows,
        rejected_worker_id=args.rejected_worker_id,
        specialist_workers=specialist_workers,
        limit=args.limit,
        min_gate_score=args.min_gate_score,
        max_universal_fail_similarity=args.max_universal_fail_similarity,
        max_pass_latency_ms=args.max_pass_latency_ms,
    )
    report = {key: value for key, value in selection.items() if key != "selected_tasks"}
    report["task_source"] = str(args.tasks)
    report["qwen_outcomes"] = str(args.qwen_outcomes)
    report["prior_outcomes"] = [str(path) for path in args.prior_outcomes]

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
