from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.conditional_policy import worker_by_id
from mempool.fallback_head import ranked_distribution
from mempool.routing_dataset import read_routing_records, validate_routing_records

try:
    from tools.evaluate_active_policy import load_active_router
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router


def fallback_record(record: dict[str, Any], router: Any) -> dict[str, Any] | None:
    ranked = ranked_distribution(record, router)
    if len(ranked) < 2:
        return None
    top_worker_id, top_probability = ranked[0]
    second_worker_id, second_probability = ranked[1]
    top_worker = worker_by_id(record, top_worker_id)
    second_worker = worker_by_id(record, second_worker_id)
    top_passed = bool(top_worker["passed"])
    second_passed = bool(second_worker["passed"])
    fallback_opportunity = not top_passed
    useful_fallback = fallback_opportunity and second_passed
    fallback_hurt = fallback_opportunity and not second_passed
    extra_latency_ms = int(second_worker["latency_ms"]) if fallback_opportunity else 0
    return {
        "task_id": record["task_id"],
        "benchmark_id": record["benchmark_id"],
        "task_family": record["task_family"],
        "prompt": record["prompt"],
        "prompt_features": record.get("prompt_features", {}),
        "top_worker_id": top_worker_id,
        "top_probability": top_probability,
        "top_passed": top_passed,
        "top_latency_ms": int(top_worker["latency_ms"]),
        "second_worker_id": second_worker_id,
        "second_probability": second_probability,
        "second_passed": second_passed,
        "second_latency_ms": int(second_worker["latency_ms"]),
        "first_second_margin": top_probability - second_probability,
        "fallback_opportunity": fallback_opportunity,
        "useful_fallback": useful_fallback,
        "fallback_hurt": fallback_hurt,
        "fallback_label": float(useful_fallback),
        "extra_latency_ms": extra_latency_ms,
        "any_worker_passed": any(bool(worker["passed"]) for worker in record["workers"]),
        "target_worker_id": record["target_worker_id"],
    }


def build_fallback_records(records: list[dict[str, Any]], router: Any) -> list[dict[str, Any]]:
    return [
        built
        for record in records
        if (built := fallback_record(record, router)) is not None
    ]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    task_count = len(records)
    opportunity_count = sum(1 for record in records if record["fallback_opportunity"])
    useful_count = sum(1 for record in records if record["useful_fallback"])
    hurt_count = sum(1 for record in records if record["fallback_hurt"])
    solvable_count = sum(1 for record in records if record["any_worker_passed"])
    return {
        "task_count": task_count,
        "fallback_opportunity_count": opportunity_count,
        "useful_fallback_count": useful_count,
        "fallback_hurt_count": hurt_count,
        "solvable_task_count": solvable_count,
        "useful_fallback_rate": useful_count / opportunity_count if opportunity_count else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build fallback-action training records from a routing dataset."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    router, active = load_active_router(args.registry)
    records = read_routing_records(args.input)
    errors = validate_routing_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    fallback_records = build_fallback_records(records, router)
    report = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "input": str(args.input),
        "output": str(args.output),
        **summarize(fallback_records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in fallback_records),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
