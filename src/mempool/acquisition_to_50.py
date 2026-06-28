from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .routing_dataset import read_routing_records


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_task_file(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"expected task list in {path}")
    return data


def active_task_ids(active_dataset: Path) -> set[str]:
    return {str(record["task_id"]) for record in read_routing_records(active_dataset)}


def unique_fresh_tasks(
    task_files: list[Path],
    exclude_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen = set(exclude_ids)
    for task_file in task_files:
        for task in read_task_file(task_file):
            task_id = str(task["id"])
            if task_id in seen:
                continue
            selected.append(task)
            seen.add(task_id)
            if len(selected) >= limit:
                return selected
    return selected


def write_task_file(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_sequence(
    *,
    run_id: str,
    tasks_path: Path,
    worker_pool: Path,
    summary_path: Path,
    outcomes_path: Path,
    audit_path: Path,
    routing_path: Path,
    merge_audit_path: Path,
    active_dataset: Path,
    merged_output: Path,
    repeat_count: int,
    required_packages: list[str],
) -> dict[str, str]:
    package_args = " ".join(f"--required-package {package}" for package in required_packages)
    converter_package_args = " ".join(
        f"--required-evaluator-package {package}" for package in required_packages
    )
    return {
        "run_repeated_eval": (
            f".venv-bigcodebench/bin/python tools/run_real_smoke_benchmark.py "
            f"--config {worker_pool} --tasks {tasks_path} --repeat-count {repeat_count} "
            f"--run-id {run_id} --output {summary_path} --outcomes {outcomes_path} "
            f"--eval-timeout-seconds 10 --resume --progress {package_args}"
        ),
        "audit_outcomes": (
            f"PYTHONPATH=src python3 tools/audit_outcome_rows.py --input {outcomes_path} "
            f"--output {audit_path} --min-workers-per-task 4 --min-samples-per-worker-task {repeat_count} "
            f"{package_args}"
        ),
        "build_routing_dataset": (
            f"PYTHONPATH=src python3 tools/build_repeated_routing_dataset.py --input {outcomes_path} "
            f"--output {routing_path} --min-samples-per-worker-task {repeat_count} "
            f"{converter_package_args}"
        ),
        "audit_merge_readiness": (
            f"PYTHONPATH=src python3 tools/audit_routing_merge_readiness.py --input {routing_path} "
            f"--output {merge_audit_path} --min-target-pass-rate 1.0"
        ),
        "guarded_merge": (
            f"PYTHONPATH=src python3 tools/merge_routing_datasets.py --input {active_dataset} {routing_path} "
            f"--output {merged_output} --require-merge-ready --min-target-pass-rate 1.0"
        ),
    }


def build_acquisition_to_50_plan(
    *,
    active_dataset: Path,
    readiness_report: Path,
    candidate_task_files: list[Path],
    output_tasks: Path,
    worker_pool: Path = Path("research/evals/ollama_cloud_worker_pool_top4.json"),
    run_id: str = "20260627-acquisition-to-50-wave1",
    min_tasks: int = 50,
    repeat_count: int = 2,
    overselect_multiplier: float = 1.5,
    required_packages: list[str] | None = None,
) -> dict[str, Any]:
    required_packages = required_packages or ["numpy", "pandas"]
    readiness = read_json(readiness_report)
    active_rows = len(read_routing_records(active_dataset))
    rows_needed = max(0, min_tasks - active_rows)
    target_batch_size = max(rows_needed, math.ceil(rows_needed * overselect_multiplier))
    exclude_ids = active_task_ids(active_dataset)
    selected_tasks = unique_fresh_tasks(candidate_task_files, exclude_ids, target_batch_size)
    write_task_file(output_tasks, selected_tasks)

    stem = Path(run_id).name
    summary_path = Path(f"research/evals/results/{stem}.json")
    outcomes_path = Path(f"research/evals/results/{stem}.jsonl")
    audit_path = Path(f"research/evals/results/{stem}_audit.json")
    routing_path = Path(f"research/datasets/{stem}-routing.jsonl")
    merge_audit_path = Path(f"research/evals/results/{stem}_merge_audit.json")
    merged_output = Path(f"research/datasets/{stem}-merged-50candidate-routing.jsonl")

    stage = {
        "id": "wave1-repeat-top4",
        "purpose": "Acquire enough stable repeated top-four BigCodeBench rows to clear the 50-task M5 gate.",
        "task_count": len(selected_tasks),
        "task_file": str(output_tasks),
        "worker_pool": str(worker_pool),
        "repeat_count": repeat_count,
        "required_packages": required_packages,
        "commands": command_sequence(
            run_id=run_id,
            tasks_path=output_tasks,
            worker_pool=worker_pool,
            summary_path=summary_path,
            outcomes_path=outcomes_path,
            audit_path=audit_path,
            routing_path=routing_path,
            merge_audit_path=merge_audit_path,
            active_dataset=active_dataset,
            merged_output=merged_output,
            repeat_count=repeat_count,
            required_packages=required_packages,
        ),
    }
    expected_calls = len(selected_tasks) * repeat_count * 4
    return {
        "plan_id": "acquisition-to-50",
        "status": "ready_to_run",
        "readiness_report": str(readiness_report),
        "readiness_decision": readiness.get("decision"),
        "active_dataset": str(active_dataset),
        "active_rows": active_rows,
        "min_tasks": min_tasks,
        "rows_needed": rows_needed,
        "overselect_multiplier": overselect_multiplier,
        "candidate_task_sources": [str(path) for path in candidate_task_files],
        "selected_task_ids": [str(task["id"]) for task in selected_tasks],
        "selected_task_count": len(selected_tasks),
        "covers_rows_needed_if_all_merge_ready": len(selected_tasks) >= rows_needed,
        "estimated_model_calls": expected_calls,
        "stages": [stage],
        "merge_policy": {
            "require_outcome_audit": True,
            "require_routing_validation": True,
            "require_merge_ready": True,
            "min_target_pass_rate": 1.0,
            "allow_all_fail_tasks": False,
        },
        "stop_condition": (
            "After a successful guarded merge, regenerate the active policy and rerun "
            "research/programs/small_orchestrator_readiness.json. Continue waves until active_rows >= 50."
        ),
    }
