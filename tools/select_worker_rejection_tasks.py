from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected task list in {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_latency_ms(row: dict[str, Any]) -> float | None:
    for key in ("latency_ms", "mean_latency_ms"):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def rejected_task_ids(
    rows: list[dict[str, Any]],
    *,
    worker_id: str,
    max_pass_latency_ms: float | None = None,
) -> list[str]:
    ids = []
    seen = set()
    for row in rows:
        if str(row.get("worker_id")) != worker_id:
            continue
        task_id = str(row["task_id"])
        if task_id in seen:
            continue
        passed = bool(row.get("passed"))
        latency = row_latency_ms(row)
        slow_pass = (
            passed
            and max_pass_latency_ms is not None
            and latency is not None
            and latency > max_pass_latency_ms
        )
        if (not passed) or slow_pass:
            seen.add(task_id)
            ids.append(task_id)
    return ids


def select_tasks(
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    worker_id: str,
    max_pass_latency_ms: float | None = None,
) -> list[dict[str, Any]]:
    ids = set(
        rejected_task_ids(
            rows,
            worker_id=worker_id,
            max_pass_latency_ms=max_pass_latency_ms,
        )
    )
    return [task for task in tasks if str(task["id"]) in ids]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select source tasks where a worker failed or was too slow."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-pass-latency-ms", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tasks = read_json(args.tasks)
    rows = read_jsonl(args.outcomes)
    selected = select_tasks(
        tasks,
        rows,
        worker_id=args.worker_id,
        max_pass_latency_ms=args.max_pass_latency_ms,
    )
    task_ids = [str(task["id"]) for task in selected]
    report = {
        "worker_id": args.worker_id,
        "max_pass_latency_ms": args.max_pass_latency_ms,
        "selected": len(selected),
        "task_ids": task_ids,
        "output": str(args.output),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
