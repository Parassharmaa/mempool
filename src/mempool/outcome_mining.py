from __future__ import annotations

from pathlib import Path
from typing import Any

from .routing_merge_filter import filter_merge_ready_records


def row_worker(worker: dict[str, Any]) -> str:
    return str(worker.get("worker_id", ""))


def is_qwen_worker(worker_id: str) -> bool:
    return "qwen" in worker_id.lower()


def stable_workers(record: dict[str, Any], min_pass_rate: float = 1.0) -> list[dict[str, Any]]:
    return [
        worker
        for worker in record.get("workers", [])
        if float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0)) >= min_pass_rate
    ]


def is_broad_pass_latency_row(record: dict[str, Any], min_pass_rate: float = 1.0) -> bool:
    workers = record.get("workers", [])
    return bool(workers) and len(stable_workers(record, min_pass_rate=min_pass_rate)) == len(workers)


def is_stable_nonqwen_target(record: dict[str, Any], min_pass_rate: float = 1.0) -> bool:
    target = str(record.get("target_worker_id", ""))
    if not target or is_qwen_worker(target):
        return False
    for worker in record.get("workers", []):
        if row_worker(worker) != target:
            continue
        return float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0)) >= min_pass_rate
    return False


def is_exclusive_stable_nonqwen_target(record: dict[str, Any], min_pass_rate: float = 1.0) -> bool:
    return is_stable_nonqwen_target(
        record,
        min_pass_rate=min_pass_rate,
    ) and not is_broad_pass_latency_row(record, min_pass_rate=min_pass_rate)


def row_summary(record: dict[str, Any], min_pass_rate: float = 1.0) -> dict[str, Any]:
    broad_pass_latency_row = is_broad_pass_latency_row(record, min_pass_rate=min_pass_rate)
    stable_nonqwen_target = is_stable_nonqwen_target(record, min_pass_rate=min_pass_rate)
    return {
        "task_id": record.get("task_id"),
        "target_worker_id": record.get("target_worker_id"),
        "target_pass_rate": next(
            (
                float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0))
                for worker in record.get("workers", [])
                if row_worker(worker) == record.get("target_worker_id")
            ),
            0.0,
        ),
        "worker_count": len(record.get("workers", [])),
        "stable_worker_count": len(stable_workers(record, min_pass_rate=min_pass_rate)),
        "broad_pass_latency_row": broad_pass_latency_row,
        "stable_nonqwen_target": stable_nonqwen_target,
        "exclusive_stable_nonqwen_target": stable_nonqwen_target and not broad_pass_latency_row,
    }


def summarize_outcome_source(
    *,
    source_path: Path,
    records: list[dict[str, Any]],
    min_pass_rate: float = 1.0,
) -> dict[str, Any]:
    merge_ready, merge_report = filter_merge_ready_records(records, min_target_pass_rate=min_pass_rate)
    row_summaries = [row_summary(record, min_pass_rate=min_pass_rate) for record in merge_ready]
    broad_pass = [item for item in row_summaries if item["broad_pass_latency_row"]]
    stable_nonqwen = [item for item in row_summaries if item["stable_nonqwen_target"]]
    exclusive_stable_nonqwen = [
        item for item in row_summaries if item["exclusive_stable_nonqwen_target"]
    ]
    score = len(exclusive_stable_nonqwen) * 4.0 + len(broad_pass) * 2.0 + len(merge_ready)
    return {
        "source": str(source_path),
        "input_records": len(records),
        "merge_ready_records": len(merge_ready),
        "dropped_records": merge_report["dropped_records"],
        "broad_pass_latency_rows": len(broad_pass),
        "stable_nonqwen_targets": len(stable_nonqwen),
        "exclusive_stable_nonqwen_targets": len(exclusive_stable_nonqwen),
        "score": round(score, 4),
        "merge_ready_task_ids": [str(record.get("task_id")) for record in merge_ready],
        "broad_pass_task_ids": [str(item["task_id"]) for item in broad_pass],
        "stable_nonqwen_task_ids": [str(item["task_id"]) for item in stable_nonqwen],
        "exclusive_stable_nonqwen_task_ids": [
            str(item["task_id"]) for item in exclusive_stable_nonqwen
        ],
        "rows": row_summaries,
    }


def rank_outcome_sources(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        summaries,
        key=lambda item: (
            -float(item["score"]),
            -int(item["exclusive_stable_nonqwen_targets"]),
            -int(item["stable_nonqwen_targets"]),
            -int(item["broad_pass_latency_rows"]),
            str(item["source"]),
        ),
    )
