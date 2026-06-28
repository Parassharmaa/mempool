from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def task_ids_from_summaries(
    summaries: list[dict[str, Any]],
    universal_min_workers: int = 1,
) -> tuple[set[str], set[str]]:
    screened: set[str] = set()
    universal_failures: set[str] = set()
    for summary in summaries:
        for item in summary.get("by_task", []):
            task_id = str(item["task_id"])
            screened.add(task_id)
            if item.get("universal_failure") and int(item.get("worker_count", 0)) >= universal_min_workers:
                universal_failures.add(task_id)
        for task_id in summary.get("universal_failure_task_ids", []):
            task_id = str(task_id)
            screened.add(task_id)
        for task_id in summary.get("candidate_task_ids", []):
            screened.add(str(task_id))
    return screened, universal_failures


def select_batch(
    tasks: list[dict[str, Any]],
    screened_task_ids: set[str],
    universal_failure_task_ids: set[str],
    batch_size: int,
    task_metadata: dict[str, dict[str, Any]] | None = None,
    max_environment_risk: int | None = None,
    allowed_task_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    excluded = screened_task_ids | universal_failure_task_ids
    selected = []
    task_metadata = task_metadata or {}
    for task in tasks:
        task_id = str(task["id"])
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        if task_id in excluded:
            continue
        if max_environment_risk is not None:
            metadata = task_metadata.get(task_id, {})
            if int(metadata.get("environment_risk", 0)) > max_environment_risk:
                continue
        selected.append(task)
        if len(selected) >= batch_size:
            break
    return selected


def task_metadata_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata = {}
    for items in manifest.get("selected_by_worker", {}).values():
        for item in items:
            metadata[str(item["task_id"])] = item
    return metadata


def task_ids_for_target_worker(manifest: dict[str, Any], worker_id: str | None) -> set[str] | None:
    if worker_id is None:
        return None
    return {
        str(item["task_id"])
        for item in manifest.get("selected_by_worker", {}).get(worker_id, [])
    }


def build_batch_manifest(
    *,
    source_manifest: dict[str, Any],
    selected_tasks: list[dict[str, Any]],
    task_metadata: dict[str, dict[str, Any]] | None,
    batch_tasks_output: Path,
    run_id: str,
    output_path: Path,
    outcomes_path: Path,
    worker_pool_path: Path,
    worker_ids_to_run: list[str] | None = None,
    repeat_count: int | None = None,
    eval_timeout_seconds: int,
    benchmark_id: str = "bigcodebench-hard-specialist-acquisition-batch",
) -> dict[str, Any]:
    repeat_count = int(repeat_count if repeat_count is not None else source_manifest["repeat_count"])
    selected_task_ids = [str(task["id"]) for task in selected_tasks]
    worker_ids = list(worker_ids_to_run or source_manifest["worker_ids_to_run"])
    task_metadata = task_metadata or {}
    return {
        "benchmark_id": benchmark_id,
        "source_benchmark_id": source_manifest["benchmark_id"],
        "source_manifest": source_manifest.get("manifest_path"),
        "run_id": run_id,
        "repeat_count": repeat_count,
        "worker_ids_to_run": worker_ids,
        "selected_task_ids": selected_task_ids,
        "selected_task_metadata": {
            task_id: task_metadata[task_id]
            for task_id in selected_task_ids
            if task_id in task_metadata
        },
        "task_count": len(selected_tasks),
        "call_count": len(selected_tasks) * len(worker_ids) * repeat_count,
        "tasks_output": str(batch_tasks_output),
        "output": str(output_path),
        "outcomes": str(outcomes_path),
        "run_command": (
            "PYTHONPATH=src python3 tools/run_real_smoke_benchmark.py "
            f"--config {worker_pool_path} "
            f"--tasks {batch_tasks_output} "
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
    parser = argparse.ArgumentParser(
        description="Plan the next bounded specialist acquisition batch."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--screening-summary", type=Path, action="append", default=[])
    parser.add_argument("--universal-min-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-environment-risk", type=int)
    parser.add_argument("--target-worker")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-pool", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int)
    parser.add_argument("--benchmark-id", default="bigcodebench-hard-specialist-acquisition-batch")
    parser.add_argument("--eval-timeout-seconds", type=int, default=20)
    parser.add_argument("--tasks-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = read_json(args.manifest)
    source_manifest["manifest_path"] = str(args.manifest)
    worker_pool = read_json(args.worker_pool)
    worker_ids_to_run = [str(worker["id"]) for worker in worker_pool.get("workers", [])]
    task_metadata = task_metadata_from_manifest(source_manifest)
    allowed_task_ids = task_ids_for_target_worker(source_manifest, args.target_worker)
    tasks = read_json(args.tasks)
    summaries = [read_json(path) for path in args.screening_summary]
    screened_task_ids, universal_failure_task_ids = task_ids_from_summaries(
        summaries,
        universal_min_workers=args.universal_min_workers,
    )
    selected_tasks = select_batch(
        tasks,
        screened_task_ids=screened_task_ids,
        universal_failure_task_ids=universal_failure_task_ids,
        batch_size=args.batch_size,
        task_metadata=task_metadata,
        max_environment_risk=args.max_environment_risk,
        allowed_task_ids=allowed_task_ids,
    )
    manifest = build_batch_manifest(
        source_manifest=source_manifest,
        selected_tasks=selected_tasks,
        task_metadata=task_metadata,
        batch_tasks_output=args.tasks_output,
        run_id=args.run_id,
        output_path=args.output,
        outcomes_path=args.outcomes,
        worker_pool_path=args.worker_pool,
        worker_ids_to_run=worker_ids_to_run or None,
        repeat_count=args.repeat_count,
        eval_timeout_seconds=args.eval_timeout_seconds,
        benchmark_id=args.benchmark_id,
    )
    manifest["screened_task_ids"] = sorted(screened_task_ids)
    manifest["universal_failure_task_ids"] = sorted(universal_failure_task_ids)
    manifest["universal_min_workers"] = args.universal_min_workers
    manifest["max_environment_risk"] = args.max_environment_risk
    manifest["target_worker"] = args.target_worker

    args.tasks_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.tasks_output.write_text(
        json.dumps(selected_tasks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
