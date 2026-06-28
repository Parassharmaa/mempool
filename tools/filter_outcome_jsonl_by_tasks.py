from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_task_ids(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected task list in {path}")
    return {str(task["id"]) for task in data}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def filter_rows(rows: list[dict[str, Any]], task_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row["task_id"] in task_ids]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter outcome JSONL rows to tasks listed in a task JSON file."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_ids = read_task_ids(args.tasks)
    rows = filter_rows(read_jsonl(args.input), task_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(rows),
                "task_ids": sorted(task_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
