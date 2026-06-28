from __future__ import annotations

from itertools import combinations
from typing import Any

from .outcome_mining import is_broad_pass_latency_row


def worker_pass_rate(worker: dict[str, Any]) -> float:
    return float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0))


def worker_latency_ms(worker: dict[str, Any]) -> float:
    return float(worker.get("latency_ms", 0.0) or 0.0)


def worker_by_id(record: dict[str, Any], worker_id: str) -> dict[str, Any] | None:
    for worker in record.get("workers", []):
        if str(worker.get("worker_id", "")) == worker_id:
            return worker
    return None


def probe_passed(
    record: dict[str, Any],
    worker_id: str,
    *,
    min_pass_rate: float = 1.0,
) -> bool:
    worker = worker_by_id(record, worker_id)
    return bool(worker) and worker_pass_rate(worker) >= min_pass_rate


def probe_latency_ms(record: dict[str, Any], worker_ids: list[str]) -> float:
    return sum(
        worker_latency_ms(worker)
        for worker_id in worker_ids
        for worker in [worker_by_id(record, worker_id)]
        if worker is not None
    )


def probe_policy_prediction(
    record: dict[str, Any],
    worker_ids: list[str],
    *,
    mode: str = "all",
    min_pass_rate: float = 1.0,
) -> bool:
    if not worker_ids:
        return False
    probe_results = [
        probe_passed(record, worker_id, min_pass_rate=min_pass_rate)
        for worker_id in worker_ids
    ]
    if mode == "all":
        return all(probe_results)
    if mode == "any":
        return any(probe_results)
    raise ValueError(f"unsupported probe policy mode: {mode}")


def confusion_metrics(
    *,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> dict[str, Any]:
    task_count = tp + fp + tn + fn
    positive_count = tp + fn
    predicted_positive_count = tp + fp
    return {
        "task_count": task_count,
        "accuracy": (tp + tn) / task_count if task_count else 0.0,
        "precision": tp / predicted_positive_count if predicted_positive_count else 0.0,
        "recall": tp / positive_count if positive_count else 0.0,
        "positive_count": positive_count,
        "predicted_positive_count": predicted_positive_count,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_probe_policy(
    records: list[dict[str, Any]],
    worker_ids: list[str],
    *,
    mode: str = "all",
    min_pass_rate: float = 1.0,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    rows = []
    total_probe_latency_ms = 0.0
    for record in records:
        label = is_broad_pass_latency_row(record, min_pass_rate=min_pass_rate)
        predicted = probe_policy_prediction(
            record,
            worker_ids,
            mode=mode,
            min_pass_rate=min_pass_rate,
        )
        latency_ms = probe_latency_ms(record, worker_ids)
        total_probe_latency_ms += latency_ms
        tp += int(predicted and label)
        fp += int(predicted and not label)
        tn += int((not predicted) and (not label))
        fn += int((not predicted) and label)
        rows.append(
            {
                "task_id": str(record.get("task_id", "")),
                "label": label,
                "predicted": predicted,
                "probe_latency_ms": latency_ms,
                "probe_results": {
                    worker_id: probe_passed(
                        record,
                        worker_id,
                        min_pass_rate=min_pass_rate,
                    )
                    for worker_id in worker_ids
                },
            }
        )

    metrics = confusion_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
    metrics.update(
        {
            "policy": "latency-safe-probe",
            "probe_worker_ids": worker_ids,
            "mode": mode,
            "min_pass_rate": min_pass_rate,
            "mean_probe_latency_ms": (
                total_probe_latency_ms / len(records) if records else 0.0
            ),
            "total_probe_latency_ms": total_probe_latency_ms,
            "rows": rows,
        }
    )
    return metrics


def record_worker_ids(records: list[dict[str, Any]]) -> list[str]:
    worker_ids = {
        str(worker.get("worker_id", ""))
        for record in records
        for worker in record.get("workers", [])
        if str(worker.get("worker_id", ""))
    }
    return sorted(worker_ids)


def sweep_probe_policies(
    records: list[dict[str, Any]],
    *,
    max_probe_count: int = 2,
    modes: list[str] | None = None,
    min_pass_rate: float = 1.0,
) -> list[dict[str, Any]]:
    worker_ids = record_worker_ids(records)
    modes = modes or ["all", "any"]
    results = []
    for count in range(1, min(max_probe_count, len(worker_ids)) + 1):
        for probe_worker_ids in combinations(worker_ids, count):
            for mode in modes:
                if count == 1 and mode == "any":
                    continue
                results.append(
                    evaluate_probe_policy(
                        records,
                        list(probe_worker_ids),
                        mode=mode,
                        min_pass_rate=min_pass_rate,
                    )
                )
    return sorted(
        results,
        key=lambda item: (
            -float(item["precision"]),
            -float(item["recall"]),
            float(item["mean_probe_latency_ms"]),
            len(item["probe_worker_ids"]),
            ",".join(item["probe_worker_ids"]),
            item["mode"],
        ),
    )
