from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.routing_dataset import read_routing_records, validate_routing_records
from mempool.routing_merge_audit import audit_routing_merge_readiness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether a routing dataset is suitable for merge into refresh training data."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-target-pass-rate", type=float, default=1.0)
    parser.add_argument("--allow-all-fail-tasks", action="store_true")
    args = parser.parse_args()

    records = read_routing_records(args.input)
    validation_errors = validate_routing_records(records)
    report = audit_routing_merge_readiness(
        records,
        min_target_pass_rate=args.min_target_pass_rate,
        allow_all_fail_tasks=args.allow_all_fail_tasks,
    )
    report["validation_errors"] = validation_errors
    if validation_errors:
        report["ready_to_merge"] = False
        report["reasons"] = [*report["reasons"], "routing dataset validation failed"]

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ready_to_merge"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
