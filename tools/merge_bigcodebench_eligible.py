from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.bigcodebench import merge_scan_reports, merge_task_lists


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge BigCodeBench eligible scan segments.")
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "evals" / "bigcodebench_hard_eligible_merged_tasks.json")
    parser.add_argument("--report", type=Path, default=ROOT / "research" / "evals" / "bigcodebench_hard_eligible_merged_report.json")
    args = parser.parse_args()

    if len(args.tasks) != len(args.reports):
        raise SystemExit("--tasks and --reports must have the same number of files")

    task_lists = [json.loads(path.read_text(encoding="utf-8")) for path in args.tasks]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    merged_tasks = merge_task_lists(task_lists)
    merged_report = merge_scan_reports(reports, merged_tasks)
    merged_report["task_inputs"] = [str(path) for path in args.tasks]
    merged_report["report_inputs"] = [str(path) for path in args.reports]
    merged_report["output"] = str(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged_tasks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(merged_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(merged_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
