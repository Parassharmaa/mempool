from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.bigcodebench import (
    BigCodeBenchSource,
    fetch_rows,
    normalize_bigcodebench_row,
    probe_canonical_solution,
    select_minipilot_tasks,
)
from mempool.smoke_benchmark import SmokeCodeBenchmarkAdapter


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a small, diverse BigCodeBench-Hard mini-pilot.")
    parser.add_argument("--input", type=Path, default=ROOT / "research" / "evals" / "bigcodebench_hard_smoke_tasks.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "evals" / "bigcodebench_hard_minipilot_tasks.json")
    parser.add_argument("--report", type=Path, default=ROOT / "research" / "evals" / "bigcodebench_hard_minipilot_report.json")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--probe-canonical", action="store_true")
    parser.add_argument("--dataset", default="bigcode/bigcodebench-hard")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="v0.1.4")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--eval-timeout-seconds", type=int, default=20)
    parser.add_argument(
        "--preferred-categories",
        nargs="+",
        default=["subprocess", "filesystem", "datasci"],
    )
    args = parser.parse_args()

    tasks = json.loads(args.input.read_text(encoding="utf-8"))
    canonical_probe = []
    eligibility = None
    if args.probe_canonical:
        rows = fetch_rows(
            BigCodeBenchSource(
                dataset=args.dataset,
                config=args.config,
                split=args.split,
                offset=args.offset,
                limit=args.limit,
            )
        )
        adapter = SmokeCodeBenchmarkAdapter(args.input, timeout_seconds=args.eval_timeout_seconds)
        eligibility = {}
        for row in rows:
            task = normalize_bigcodebench_row(row)
            probe = probe_canonical_solution(row, adapter=adapter)
            eligibility[task.id] = probe.passed
            canonical_probe.append(
                {
                    "task_id": task.id,
                    "passed": probe.passed,
                    "failure_mode": probe.failure_mode,
                    "stderr_tail": probe.stderr_tail,
                }
            )

    selection = select_minipilot_tasks(
        tasks,
        count=args.count,
        preferred_categories=tuple(args.preferred_categories),
        eligibility=eligibility,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection["selected_tasks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "count": args.count,
        "preferred_categories": args.preferred_categories,
        "canonical_probe": canonical_probe,
        "selected_task_ids": selection["selected_task_ids"],
        "selection_reasons": selection["selection_reasons"],
        "analysis": selection["analysis"],
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
