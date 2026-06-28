from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.routing_dataset import validate_routing_records
from mempool.routing_merge_audit import audit_routing_merge_readiness


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def merge_records(paths: list[Path]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in read_jsonl(path):
            by_task[record["task_id"]] = record
    return [by_task[task_id] for task_id in sorted(by_task)]


def assert_merge_ready(
    records: list[dict[str, Any]],
    min_target_pass_rate: float = 1.0,
    allow_all_fail_tasks: bool = False,
) -> dict[str, Any]:
    validation_errors = validate_routing_records(records)
    report = audit_routing_merge_readiness(
        records,
        min_target_pass_rate=min_target_pass_rate,
        allow_all_fail_tasks=allow_all_fail_tasks,
    )
    report["validation_errors"] = validation_errors
    if validation_errors:
        report["ready_to_merge"] = False
        report["reasons"] = [*report["reasons"], "routing dataset validation failed"]
    if not report["ready_to_merge"]:
        raise ValueError("routing dataset is not merge-ready: " + "; ".join(report["reasons"]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge routing dataset JSONL files.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-merge-ready", action="store_true")
    parser.add_argument("--min-target-pass-rate", type=float, default=1.0)
    parser.add_argument("--allow-all-fail-tasks", action="store_true")
    args = parser.parse_args()

    records = merge_records(args.input)
    merge_audit = None
    if args.require_merge_ready:
        try:
            merge_audit = assert_merge_ready(
                records,
                min_target_pass_rate=args.min_target_pass_rate,
                allow_all_fail_tasks=args.allow_all_fail_tasks,
            )
        except ValueError as error:
            print(json.dumps({"error": str(error), "records": len(records)}, indent=2))
            return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in args.input],
                "output": str(args.output),
                "records": len(records),
                "task_ids": [record["task_id"] for record in records],
                "merge_audit": merge_audit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
