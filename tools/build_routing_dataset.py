from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def softmax(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    scaled = [score / temperature for score in scores]
    offset = max(scaled)
    exps = [math.exp(score - offset) for score in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def normalized_latency_penalty(latency_ms: int, max_latency_ms: int) -> float:
    if max_latency_ms <= 0:
        return 0.0
    return latency_ms / max_latency_ms


def reward_for(row: dict[str, Any], max_latency_ms: int, latency_weight: float) -> float:
    solved = 1.0 if row["passed"] else 0.0
    latency_penalty = normalized_latency_penalty(int(row["latency_ms"]), max_latency_ms)
    return solved - latency_weight * latency_penalty - float(row.get("cost_usd") or 0.0)


def prompt_features_for(row: dict[str, Any]) -> dict[str, Any]:
    analysis = classify_task(
        {
            "id": row["task_id"],
            "prompt": row["prompt"],
            "tests": [],
        }
    )
    return {
        "length_chars": len(row["prompt"]),
        "requires_code": True,
        "requires_tools": False,
        "libraries": analysis["libraries"],
        "missing_libraries": analysis["missing_libraries"],
        "categories": analysis["categories"],
        "primary_category": analysis["primary_category"],
        "environment_risk": analysis["environment_risk"],
        "plausibility_score": analysis["plausibility_score"],
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
    rows: list[dict[str, Any]],
    temperature: float = 0.25,
    latency_weight: float = 0.05,
    required_evaluator_packages: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows, _ = filter_rows_by_evaluator_packages(
        rows,
        required_evaluator_packages or [],
    )
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)

    records = []
    for task_id in sorted(by_task):
        task_rows = sorted(by_task[task_id], key=lambda row: row["worker_id"])
        max_latency_ms = max(int(row["latency_ms"]) for row in task_rows)
        rewards = [
            reward_for(row, max_latency_ms=max_latency_ms, latency_weight=latency_weight)
            for row in task_rows
        ]
        target_probs = softmax(rewards, temperature=temperature)
        workers = []
        for row, reward, probability in zip(task_rows, rewards, target_probs, strict=True):
            workers.append(
                {
                    "worker_id": row["worker_id"],
                    "model": row["model"],
                    "passed": row["passed"],
                    "score": row["score"],
                    "latency_ms": row["latency_ms"],
                    "cost_usd": row["cost_usd"],
                    "failure_mode": row["failure_mode"],
                    "reward": round(reward, 6),
                    "target_probability": round(probability, 6),
                }
            )

        best = max(workers, key=lambda worker: worker["target_probability"])
        template = task_rows[0]
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
    parser = argparse.ArgumentParser(description="Build routing dataset from outcome JSONL.")
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
