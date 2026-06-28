from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["task_id"], row["worker_id"])].append(row)

    records = []
    for (task_id, worker_id), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row.get("sample_index", 0)))
        attempts = len(ordered)
        solved = sum(1 for row in ordered if row["passed"])
        latencies = [int(row["latency_ms"]) for row in ordered]
        failure_modes: dict[str, int] = defaultdict(int)
        for row in ordered:
            failure_modes[str(row["failure_mode"])] += 1
        records.append(
            {
                "task_id": task_id,
                "worker_id": worker_id,
                "model": ordered[0]["model"],
                "attempts": attempts,
                "solved": solved,
                "pass_rate": solved / attempts if attempts else 0.0,
                "mean_latency_ms": round(sum(latencies) / attempts, 2) if attempts else None,
                "sample_passes": [bool(row["passed"]) for row in ordered],
                "failure_modes": dict(sorted(failure_modes.items())),
            }
        )

    by_worker = []
    worker_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        worker_groups[record["worker_id"]].append(record)
        task_groups[record["task_id"]].append(record)
    for worker_id, worker_records in sorted(worker_groups.items()):
        attempts = sum(int(record["attempts"]) for record in worker_records)
        solved = sum(int(record["solved"]) for record in worker_records)
        weighted_latency = sum(
            float(record["mean_latency_ms"]) * int(record["attempts"])
            for record in worker_records
            if record["mean_latency_ms"] is not None
        )
        by_worker.append(
            {
                "worker_id": worker_id,
                "task_count": len(worker_records),
                "attempts": attempts,
                "solved": solved,
                "pass_rate": solved / attempts if attempts else 0.0,
                "mean_latency_ms": round(weighted_latency / attempts, 2) if attempts else None,
            }
        )

    by_task = []
    universal_failure_task_ids = []
    candidate_task_ids = []
    for task_id, task_records in sorted(task_groups.items()):
        attempts = sum(int(record["attempts"]) for record in task_records)
        solved = sum(int(record["solved"]) for record in task_records)
        best_record = max(
            task_records,
            key=lambda record: (
                float(record["pass_rate"]),
                -float(record["mean_latency_ms"] or 1_000_000_000),
                str(record["worker_id"]),
            ),
        )
        task_summary = {
            "task_id": task_id,
            "worker_count": len(task_records),
            "attempts": attempts,
            "solved": solved,
            "universal_failure": solved == 0,
            "candidate_for_conversion": solved > 0,
            "best_worker_id": best_record["worker_id"] if solved > 0 else None,
            "best_pass_rate": best_record["pass_rate"] if solved > 0 else 0.0,
            "best_mean_latency_ms": best_record["mean_latency_ms"] if solved > 0 else None,
        }
        by_task.append(task_summary)
        if task_summary["universal_failure"]:
            universal_failure_task_ids.append(task_id)
        if task_summary["candidate_for_conversion"]:
            candidate_task_ids.append(task_id)

    return {
        "records": records,
        "by_worker": by_worker,
        "by_task": by_task,
        "record_count": len(records),
        "outcome_count": len(rows),
        "task_count": len(task_groups),
        "candidate_task_ids": candidate_task_ids,
        "universal_failure_task_ids": universal_failure_task_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize repeated outcome JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = summarize(read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **{k: report[k] for k in ("record_count", "outcome_count")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
