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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def annotate_weights(
    records: list[dict[str, Any]],
    *,
    default_weight: float,
    task_weights: dict[str, float],
) -> list[dict[str, Any]]:
    annotated = []
    for record in records:
        updated = dict(record)
        task_id = str(record.get("task_id", ""))
        updated["training_weight"] = float(task_weights.get(task_id, default_weight))
        annotated.append(updated)
    return annotated


def read_task_ids(path: Path) -> set[str]:
    return {
        str(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotate routing records with optional per-row training weights."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-weight", type=float, default=1.0)
    parser.add_argument(
        "--task-id-weight-file",
        action="append",
        nargs=2,
        metavar=("TASK_IDS", "WEIGHT"),
        default=[],
        help="Apply WEIGHT to task ids listed one per line in TASK_IDS.",
    )
    args = parser.parse_args()

    task_weights: dict[str, float] = {}
    for task_file, weight_text in args.task_id_weight_file:
        weight = float(weight_text)
        for task_id in read_task_ids(Path(task_file)):
            task_weights[task_id] = weight

    records = annotate_weights(
        read_jsonl(args.input),
        default_weight=args.default_weight,
        task_weights=task_weights,
    )
    write_jsonl(args.output, records)
    weights: dict[str, int] = {}
    for record in records:
        weight = str(record["training_weight"])
        weights[weight] = weights.get(weight, 0) + 1
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "records": len(records),
                "weight_counts": weights,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
