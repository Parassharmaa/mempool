from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mempool.bigcodebench import (
    BigCodeBenchSource,
    classify_task,
    materialize_tasks,
    scan_canonical_pass_rows,
)
from mempool.smoke_benchmark import SmokeCodeBenchmarkAdapter


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan BigCodeBench-Hard for locally runnable canonical-pass tasks.")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "evals" / "bigcodebench_hard_eligible_tasks.json")
    parser.add_argument("--report", type=Path, default=ROOT / "research" / "evals" / "bigcodebench_hard_eligible_report.json")
    parser.add_argument("--dataset", default="bigcode/bigcodebench-hard")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="v0.1.4")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--target-passes", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--eval-timeout-seconds", type=int, default=20)
    parser.add_argument("--mode", choices=("instruct", "complete"), default="instruct")
    args = parser.parse_args()

    adapter = SmokeCodeBenchmarkAdapter(args.output, timeout_seconds=args.eval_timeout_seconds)
    scan = scan_canonical_pass_rows(
        source=BigCodeBenchSource(
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            offset=args.offset,
            limit=args.page_size,
        ),
        adapter=adapter,
        target_passes=args.target_passes,
        page_size=args.page_size,
        max_rows=args.max_rows,
        mode=args.mode,
    )
    tasks = materialize_tasks(scan["passed_rows"], mode=args.mode)
    analyses = [classify_task(task) for task in tasks]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    probe_rows = [asdict(probe) for probe in scan["probes"]]
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "mode": args.mode,
        "offset": args.offset,
        "target_passes": args.target_passes,
        "max_rows": args.max_rows,
        "scanned": scan["scanned"],
        "next_offset": scan["next_offset"],
        "eligible_count": len(tasks),
        "output": str(args.output),
        "selected_task_ids": [task["id"] for task in tasks],
        "analysis": analyses,
        "canonical_probe": probe_rows,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
