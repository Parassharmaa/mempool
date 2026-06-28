from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.routing_dataset import read_routing_records

try:
    from tools.plan_solvability_aware_specialist_acquisition import read_screened_task_ids
    from tools.select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
except ModuleNotFoundError:
    from plan_solvability_aware_specialist_acquisition import read_screened_task_ids
    from select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def routing_task_ids(paths: list[Path]) -> set[str]:
    task_ids = set()
    for path in paths:
        task_ids.update(str(record["task_id"]) for record in read_routing_records(path))
    return task_ids


def select_direct_mining_batch(
    tasks: list[dict[str, Any]],
    *,
    exclude_task_ids: set[str],
    limit: int,
    max_environment_risk: int,
    preferred_categories: list[str] | None = None,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    preferred_categories = preferred_categories or ["filesystem", "datasci", "plotting", "subprocess", "general"]
    preferred_set = set(preferred_categories)
    candidates = []
    fresh_candidate_count = 0
    for task in tasks:
        task_id = str(task["id"])
        if task_id in exclude_task_ids:
            continue
        fresh_candidate_count += 1
        analysis = classify_task(task)
        if int(analysis["environment_risk"]) > max_environment_risk:
            continue
        categories = {str(value).lower() for value in analysis["categories"]}
        preferred_overlap = len(categories & preferred_set)
        candidates.append(
            {
                "task": task,
                "analysis": analysis,
                "preferred_overlap": preferred_overlap,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -int(item["preferred_overlap"]),
            int(item["analysis"]["environment_risk"]),
            float(item["analysis"]["plausibility_score"]),
            str(item["task"]["id"]),
        ),
    )
    selected = ranked[:limit]
    return {
        "candidate_count": fresh_candidate_count,
        "eligible_candidate_count": len(candidates),
        "excluded_count": len(exclude_task_ids),
        "max_environment_risk": max_environment_risk,
        "preferred_categories": preferred_categories,
        "selected_task_ids": [str(item["task"]["id"]) for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [report_item(item) for item in selected],
        "ranked_candidates": [report_item(item) for item in ranked],
    }


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    analysis = item["analysis"]
    return {
        "task_id": item["task"]["id"],
        "categories": analysis["categories"],
        "libraries": analysis["libraries"],
        "environment_risk": analysis["environment_risk"],
        "plausibility_score": analysis["plausibility_score"],
        "preferred_overlap": item["preferred_overlap"],
    }


def build_run_manifest(
    *,
    selection: dict[str, Any],
    worker_pool: dict[str, Any],
    worker_pool_path: Path,
    tasks_output: Path,
    run_id: str,
    output_path: Path,
    outcomes_path: Path,
    repeat_count: int,
    eval_timeout_seconds: int,
) -> dict[str, Any]:
    worker_ids = [str(worker["id"]) for worker in worker_pool.get("workers", [])]
    return {
        "benchmark_id": "bigcodebench-hard-direct-specialist-mining",
        "purpose": "Mine actual one-sample specialist positives from a fresh low-risk task pool before full comparison.",
        "run_id": run_id,
        "worker_ids_to_run": worker_ids,
        "repeat_count": repeat_count,
        "task_count": len(selection["selected_task_ids"]),
        "call_count": len(selection["selected_task_ids"]) * len(worker_ids) * repeat_count,
        "tasks_output": str(tasks_output),
        "output": str(output_path),
        "outcomes": str(outcomes_path),
        "run_command": (
            "PYTHONPATH=src python3 tools/run_real_smoke_benchmark.py "
            f"--config {worker_pool_path} "
            f"--tasks {tasks_output} "
            f"--repeat-count {repeat_count} "
            f"--eval-timeout-seconds {eval_timeout_seconds} "
            f"--run-id {run_id} "
            f"--output {output_path} "
            f"--outcomes {outcomes_path} "
            "--resume --progress --quiet"
        ),
        "summary_command": (
            "PYTHONPATH=src python3 tools/summarize_repeated_outcomes.py "
            f"--input {outcomes_path} "
            f"--output {outcomes_path.with_name(outcomes_path.stem + '-summary.json')}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan a direct one-worker specialist mining batch.")
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--exclude-screening-summary", type=Path, action="append", default=[])
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--preferred-category", action="append", default=[])
    parser.add_argument("--max-environment-risk", type=int, default=1)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--worker-pool", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--eval-timeout-seconds", type=int, default=20)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    args = parser.parse_args()

    exclude_ids = set(args.exclude_task_id)
    exclude_ids.update(routing_task_ids(args.exclude_routing_dataset))
    exclude_ids.update(read_outcome_task_ids(args.exclude_outcomes))
    for path in args.exclude_screening_summary:
        exclude_ids.update(read_screened_task_ids(path))
    tasks = load_task_union(args.tasks)
    worker_pool = read_json(args.worker_pool)
    selection = select_direct_mining_batch(
        tasks,
        exclude_task_ids=exclude_ids,
        limit=args.limit,
        max_environment_risk=args.max_environment_risk,
        preferred_categories=args.preferred_category or None,
    )
    manifest = {
        **{key: value for key, value in selection.items() if key != "selected_tasks"},
        **build_run_manifest(
            selection=selection,
            worker_pool=worker_pool,
            worker_pool_path=args.worker_pool,
            tasks_output=args.output,
            run_id=args.run_id,
            output_path=args.run_output,
            outcomes_path=args.outcomes,
            repeat_count=args.repeat_count,
            eval_timeout_seconds=args.eval_timeout_seconds,
        ),
        "task_sources": [str(path) for path in args.tasks],
        "excluded_routing_datasets": [str(path) for path in args.exclude_routing_dataset],
        "excluded_outcomes": [str(path) for path in args.exclude_outcomes],
        "excluded_screening_summaries": [str(path) for path in args.exclude_screening_summary],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection["selected_tasks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
