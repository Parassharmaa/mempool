from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mempool.conditional_policy import worker_by_id
from mempool.fallback_head import ranked_distribution
from mempool.routing_dataset import read_routing_records, validate_routing_records

try:
    from tools.evaluate_active_policy import load_active_router
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router


def _worker_rank(ranked: list[tuple[str, float]], worker_id: str) -> int:
    for index, (ranked_worker_id, _) in enumerate(ranked, start=1):
        if ranked_worker_id == worker_id:
            return index
    raise KeyError(worker_id)


def _worker_probability(ranked: list[tuple[str, float]], worker_id: str) -> float:
    for ranked_worker_id, probability in ranked:
        if ranked_worker_id == worker_id:
            return float(probability)
    raise KeyError(worker_id)


def mine_record(
    record: dict[str, Any],
    router: Any,
    source_dataset: str,
) -> dict[str, Any] | None:
    available_worker_ids = {str(worker["worker_id"]) for worker in record.get("workers", [])}
    ranked = [
        (worker_id, probability)
        for worker_id, probability in ranked_distribution(record, router)
        if worker_id in available_worker_ids
    ]
    if len(ranked) < 2:
        return None

    top_worker_id, top_probability = ranked[0]
    second_worker_id, second_probability = ranked[1]
    top_worker = worker_by_id(record, top_worker_id)
    if bool(top_worker["passed"]):
        return None

    second_worker = worker_by_id(record, second_worker_id)
    passed_alternates = [
        worker_by_id(record, worker_id)
        for worker_id, _ in ranked[1:]
        if bool(worker_by_id(record, worker_id)["passed"])
    ]
    best_ranked_alternate = passed_alternates[0] if passed_alternates else None
    fastest_passed_alternate = (
        min(passed_alternates, key=lambda worker: int(worker["latency_ms"]))
        if passed_alternates
        else None
    )
    all_alternates = []
    for worker_id, probability in ranked[1:]:
        worker = worker_by_id(record, worker_id)
        all_alternates.append(
            {
                "worker_id": worker_id,
                "rank": _worker_rank(ranked, worker_id),
                "probability": probability,
                "passed": bool(worker["passed"]),
                "latency_ms": int(worker["latency_ms"]),
                "reward": float(worker.get("reward", 0.0)),
            }
        )

    best_alternate_worker_id = (
        str(best_ranked_alternate["worker_id"]) if best_ranked_alternate else None
    )
    fastest_alternate_worker_id = (
        str(fastest_passed_alternate["worker_id"]) if fastest_passed_alternate else None
    )
    best_alternate_latency_ms = (
        int(best_ranked_alternate["latency_ms"]) if best_ranked_alternate else None
    )
    fastest_alternate_latency_ms = (
        int(fastest_passed_alternate["latency_ms"]) if fastest_passed_alternate else None
    )
    top_latency_ms = int(top_worker["latency_ms"])

    return {
        "task_id": record["task_id"],
        "benchmark_id": record["benchmark_id"],
        "task_family": record["task_family"],
        "prompt": record["prompt"],
        "prompt_features": record.get("prompt_features", {}),
        "source_dataset": source_dataset,
        "top_worker_id": top_worker_id,
        "top_probability": top_probability,
        "top_passed": False,
        "top_latency_ms": top_latency_ms,
        "second_worker_id": second_worker_id,
        "second_probability": second_probability,
        "second_passed": bool(second_worker["passed"]),
        "second_latency_ms": int(second_worker["latency_ms"]),
        "first_second_margin": top_probability - second_probability,
        "fallback_opportunity": True,
        "useful_second_fallback": bool(second_worker["passed"]),
        "useful_any_fallback": bool(best_ranked_alternate),
        "hard_negative": best_ranked_alternate is None,
        "best_ranked_alternate_worker_id": best_alternate_worker_id,
        "best_ranked_alternate_rank": (
            _worker_rank(ranked, best_alternate_worker_id) if best_alternate_worker_id else None
        ),
        "best_ranked_alternate_probability": (
            _worker_probability(ranked, best_alternate_worker_id)
            if best_alternate_worker_id
            else None
        ),
        "best_ranked_alternate_latency_ms": best_alternate_latency_ms,
        "additional_latency_to_best_ranked_alternate_ms": best_alternate_latency_ms,
        "total_latency_to_best_ranked_alternate_ms": (
            top_latency_ms + best_alternate_latency_ms
            if best_alternate_latency_ms is not None
            else None
        ),
        "fastest_passed_alternate_worker_id": fastest_alternate_worker_id,
        "fastest_passed_alternate_latency_ms": fastest_alternate_latency_ms,
        "additional_latency_to_fastest_passed_alternate_ms": fastest_alternate_latency_ms,
        "total_latency_to_fastest_passed_alternate_ms": (
            top_latency_ms + fastest_alternate_latency_ms
            if fastest_alternate_latency_ms is not None
            else None
        ),
        "target_worker_id": record["target_worker_id"],
        "target_worker_passed": bool(
            worker_by_id(record, record["target_worker_id"])["passed"]
        ),
        "solvable_by_any_worker": any(bool(worker["passed"]) for worker in record["workers"]),
        "alternate_count": len(ranked) - 1,
        "passed_alternate_count": len(passed_alternates),
        "alternates": all_alternates,
    }


def mine_datasets(
    datasets: list[Path],
    router: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    mined = []
    errors = []
    for dataset in datasets:
        records = read_routing_records(dataset)
        dataset_errors = validate_routing_records(records)
        if dataset_errors:
            errors.extend(f"{dataset}: {error}" for error in dataset_errors)
            continue
        for record in records:
            mined_record = mine_record(record, router, str(dataset))
            if mined_record is not None:
                mined.append(mined_record)
    return mined, errors


def summarize(records: list[dict[str, Any]], datasets: list[Path]) -> dict[str, Any]:
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_top_worker = Counter(record["top_worker_id"] for record in records)
    positive_by_alternate = Counter(
        record["best_ranked_alternate_worker_id"]
        for record in records
        if record["useful_any_fallback"]
    )
    unique_task_ids = sorted({record["task_id"] for record in records})
    positive_task_ids = sorted(
        {
            record["task_id"]
            for record in records
            if record["useful_any_fallback"]
        }
    )
    hard_negative_task_ids = sorted(
        {
            record["task_id"]
            for record in records
            if record["hard_negative"]
        }
    )
    for record in records:
        source = record["source_dataset"]
        by_source[source]["fallback_opportunities"] += 1
        by_source[source]["useful_second_fallbacks"] += int(record["useful_second_fallback"])
        by_source[source]["useful_any_fallbacks"] += int(record["useful_any_fallback"])
        by_source[source]["hard_negatives"] += int(record["hard_negative"])

    return {
        "datasets": [str(dataset) for dataset in datasets],
        "records_emitted": len(records),
        "unique_task_count": len(unique_task_ids),
        "fallback_opportunities": len(records),
        "useful_second_fallbacks": sum(
            1 for record in records if record["useful_second_fallback"]
        ),
        "useful_any_fallbacks": sum(1 for record in records if record["useful_any_fallback"]),
        "hard_negatives": sum(1 for record in records if record["hard_negative"]),
        "solvable_fallback_opportunities": sum(
            1 for record in records if record["solvable_by_any_worker"]
        ),
        "useful_any_fallback_rate": (
            sum(1 for record in records if record["useful_any_fallback"]) / len(records)
            if records
            else 0.0
        ),
        "positive_task_ids": positive_task_ids,
        "hard_negative_task_ids": hard_negative_task_ids,
        "by_source": {source: dict(counts) for source, counts in sorted(by_source.items())},
        "top_worker_failure_counts": dict(sorted(by_top_worker.items())),
        "positive_best_alternate_counts": dict(sorted(positive_by_alternate.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine historical active-router top-fail/alternate-pass fallback cases."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    router, active = load_active_router(args.registry)
    mined, errors = mine_datasets(args.dataset, router)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    report = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "output": str(args.output),
        **summarize(mined, args.dataset),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in mined),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
