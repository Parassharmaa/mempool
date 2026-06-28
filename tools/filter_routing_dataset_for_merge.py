from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.routing_dataset import validate_routing_records
from mempool.routing_merge_filter import filter_merge_ready_records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter routing records to merge-ready rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-target-pass-rate", type=float, default=1.0)
    parser.add_argument("--allow-all-fail-tasks", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    kept, report = filter_merge_ready_records(
        records,
        min_target_pass_rate=args.min_target_pass_rate,
        allow_all_fail_tasks=args.allow_all_fail_tasks,
    )
    report["validation_errors"] = validate_routing_records(kept)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if kept and not report["validation_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
