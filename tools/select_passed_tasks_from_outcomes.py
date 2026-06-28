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


def passed_task_ids(rows: list[dict[str, Any]], worker_id: str | None = None) -> list[str]:
    ids = []
    seen = set()
    for row in rows:
        if worker_id and row["worker_id"] != worker_id:
            continue
        if not row["passed"]:
            continue
        task_id = row["task_id"]
        if task_id in seen:
            continue
        seen.add(task_id)
        ids.append(task_id)
    return ids


def select_tasks(
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    worker_id: str | None = None,
) -> list[dict[str, Any]]:
    ids = set(passed_task_ids(rows, worker_id=worker_id))
    return [task for task in tasks if task["id"] in ids]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select source tasks that passed in an outcome JSONL."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-id")
    args = parser.parse_args()

    tasks = read_json(args.tasks)
    rows = read_jsonl(args.outcomes)
    selected = select_tasks(tasks, rows, worker_id=args.worker_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected": len(selected),
                "output": str(args.output),
                "worker_id": args.worker_id,
                "task_ids": [task["id"] for task in selected],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
