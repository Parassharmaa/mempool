from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.acquisition_to_50 import build_acquisition_to_50_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a staged BigCodeBench acquisition plan to grow the active routing dataset to 50 rows."
    )
    parser.add_argument(
        "--active-dataset",
        type=Path,
        default=Path("research/datasets/20260627-mixed-winner-23task-heldout-hard-routing.jsonl"),
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=Path("research/programs/small_orchestrator_readiness.json"),
    )
    parser.add_argument(
        "--candidate-task-file",
        type=Path,
        action="append",
        default=[],
        help="Ordered task files to draw from. Earlier files are preferred.",
    )
    parser.add_argument(
        "--worker-pool",
        type=Path,
        default=Path("research/evals/ollama_cloud_worker_pool_top4.json"),
    )
    parser.add_argument(
        "--output-tasks",
        type=Path,
        default=Path("research/evals/bigcodebench_hard_acquisition_to_50_wave1_tasks.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/programs/acquisition_to_50_plan.json"),
    )
    parser.add_argument("--run-id", default="20260627-acquisition-to-50-wave1")
    parser.add_argument("--min-tasks", type=int, default=50)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--overselect-multiplier", type=float, default=1.5)
    parser.add_argument("--required-package", action="append", default=[])
    args = parser.parse_args()

    candidate_task_files = args.candidate_task_file or [
        Path("research/evals/bigcodebench_hard_top4_specialist_fresh_batch8_tasks.json"),
        Path("research/evals/bigcodebench_hard_top4_hard_fresh_batch8_tasks.json"),
        Path("research/evals/bigcodebench_hard_top4_fresh_batch8_tasks.json"),
        Path("research/evals/bigcodebench_hard_eligible_offset150_tasks.json"),
        Path("research/evals/bigcodebench_hard_top4_offset125_tasks.json"),
        Path("research/evals/bigcodebench_hard_top4_offset99_tasks.json"),
        Path("research/evals/bigcodebench_hard_top4_offset44_full_tasks.json"),
        Path("research/evals/bigcodebench_hard_top4_offset44_tasks.json"),
        Path("research/evals/bigcodebench_hard_top4_offset0_tasks.json"),
        Path("research/evals/bigcodebench_hard_eligible_merged_tasks.json"),
    ]
    plan = build_acquisition_to_50_plan(
        active_dataset=args.active_dataset,
        readiness_report=args.readiness_report,
        candidate_task_files=candidate_task_files,
        output_tasks=args.output_tasks,
        worker_pool=args.worker_pool,
        run_id=args.run_id,
        min_tasks=args.min_tasks,
        repeat_count=args.repeat_count,
        overselect_multiplier=args.overselect_multiplier,
        required_packages=args.required_package or ["numpy", "pandas"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
