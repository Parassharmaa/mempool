from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def merge_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        for row in read_jsonl(path):
            key = (
                row["run_id"],
                row["worker_id"],
                row["task_id"],
                int(row.get("sample_index", 0)),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["task_id"],
            row["worker_id"],
            int(row.get("sample_index", 0)),
            row["run_id"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge outcome JSONL files.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = merge_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in args.input],
                "output": str(args.output),
                "rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
