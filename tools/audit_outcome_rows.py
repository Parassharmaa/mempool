from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.outcome_audit import audit_outcome_rows


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit outcome JSONL rows before routing-dataset conversion."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--required-evaluator-package", action="append", default=[])
    parser.add_argument("--min-workers-per-task", type=int, default=1)
    parser.add_argument("--min-samples-per-worker-task", type=int, default=1)
    args = parser.parse_args()

    report = audit_outcome_rows(
        read_jsonl(args.input),
        required_evaluator_packages=args.required_evaluator_package,
        min_workers_per_task=args.min_workers_per_task,
        min_samples_per_worker_task=args.min_samples_per_worker_task,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ready_for_conversion"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
