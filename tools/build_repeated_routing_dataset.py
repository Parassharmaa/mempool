from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
try:
    from tools.summarize_repeated_outcomes import read_jsonl, summarize
except ModuleNotFoundError:
    from summarize_repeated_outcomes import read_jsonl, summarize


def softmax(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    scaled = [score / temperature for score in scores]
    offset = max(scaled)
    exps = [math.exp(score - offset) for score in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def prompt_features_for(row: dict[str, Any]) -> dict[str, Any]:
    analysis = classify_task({"id": row["task_id"], "prompt": row["prompt"], "tests": []})
    evaluator_packages = row.get("evaluator_required_packages")
    missing_libraries = analysis["missing_libraries"]
    if isinstance(evaluator_packages, dict):
        missing_libraries = [
            library
            for library in analysis["libraries"]
            if library in evaluator_packages and not bool(evaluator_packages.get(library))
        ] + [
            library
            for library in analysis["missing_libraries"]
            if library not in evaluator_packages
        ]
    environment_risk = analysis["environment_risk"] - len(analysis["missing_libraries"]) + len(missing_libraries)
    plausibility_score = analysis["plausibility_score"] - 1.5 * len(analysis["missing_libraries"]) + 1.5 * len(missing_libraries)
    return {
        "length_chars": len(row["prompt"]),
        "requires_code": True,
        "requires_tools": False,
        "libraries": analysis["libraries"],
        "missing_libraries": missing_libraries,
        "categories": analysis["categories"],
        "primary_category": analysis["primary_category"],
        "environment_risk": environment_risk,
        "plausibility_score": round(plausibility_score, 4),
    }


def row_has_required_packages(row: dict[str, Any], required_packages: list[str]) -> bool:
    packages = row.get("evaluator_required_packages", {})
    if not required_packages:
        return True
    if not isinstance(packages, dict):
        return False
    return all(bool(packages.get(package)) for package in required_packages)


def filter_rows_by_evaluator_packages(
    rows: list[dict[str, Any]],
    required_packages: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    skipped = []
    for row in rows:
        if row_has_required_packages(row, required_packages):
            kept.append(row)
        else:
            skipped.append(row)
    return kept, skipped


def build_records(
    outcome_rows: list[dict[str, Any]],
    temperature: float = 0.25,
    latency_weight: float = 0.05,
    required_evaluator_packages: list[str] | None = None,
) -> list[dict[str, Any]]:
    outcome_rows, _ = filter_rows_by_evaluator_packages(
        outcome_rows,
        required_evaluator_packages or [],
    )
    summary = summarize(outcome_rows)
    prompt_by_task = {row["task_id"]: row for row in outcome_rows}
    records_by_task: dict[str, list[dict[str, Any]]] = {}
    for record in summary["records"]:
        records_by_task.setdefault(record["task_id"], []).append(record)

    records = []
    for task_id, worker_records in sorted(records_by_task.items()):
        max_latency = max(float(record["mean_latency_ms"] or 0.0) for record in worker_records)
        workers = []
        rewards = []
        for record in sorted(worker_records, key=lambda item: item["worker_id"]):
            latency_penalty = (
                float(record["mean_latency_ms"] or 0.0) / max_latency
                if max_latency > 0
                else 0.0
            )
            reward = float(record["pass_rate"]) - latency_weight * latency_penalty
            pass_rate = float(record["pass_rate"])
            if pass_rate == 1.0:
                failure_mode = None
            elif pass_rate == 0.0:
                failure_mode = "all_samples_failed"
            else:
                failure_mode = "mixed_samples"
            rewards.append(reward)
            workers.append(
                {
                    "worker_id": record["worker_id"],
                    "model": record["model"],
                    "passed": pass_rate > 0.0,
                    "score": pass_rate,
                    "latency_ms": int(round(float(record["mean_latency_ms"] or 0.0))),
                    "cost_usd": 0.0,
                    "failure_mode": failure_mode,
                    "attempts": record["attempts"],
                    "solved": record["solved"],
                    "pass_rate": pass_rate,
                    "mean_latency_ms": record["mean_latency_ms"],
                    "sample_passes": record["sample_passes"],
                    "failure_modes": record["failure_modes"],
                    "reward": round(reward, 6),
                }
            )

        probabilities = softmax(rewards, temperature=temperature)
        for worker, probability in zip(workers, probabilities, strict=True):
            worker["target_probability"] = round(probability, 6)

        best = max(workers, key=lambda worker: worker["target_probability"])
        template = prompt_by_task[task_id]
        records.append(
            {
                "task_id": task_id,
                "benchmark_id": template["benchmark_id"],
                "task_family": template["task_family"],
                "prompt": template["prompt"],
                "prompt_features": prompt_features_for(template),
                "workers": workers,
                "target_worker_id": best["worker_id"],
                "target_distribution": {
                    worker["worker_id"]: worker["target_probability"]
                    for worker in workers
                },
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build routing dataset from repeated outcome samples."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--latency-weight", type=float, default=0.05)
    parser.add_argument("--required-evaluator-package", action="append", default=[])
    args = parser.parse_args()

    outcome_rows = read_jsonl(args.input)
    filtered_rows, skipped_rows = filter_rows_by_evaluator_packages(
        outcome_rows,
        args.required_evaluator_package,
    )
    records = build_records(
        filtered_rows,
        temperature=args.temperature,
        latency_weight=args.latency_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "input_rows": len(outcome_rows),
                "skipped_rows": len(skipped_rows),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
