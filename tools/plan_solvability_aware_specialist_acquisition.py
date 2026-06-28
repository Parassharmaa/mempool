from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.acquisition_plan import build_solvability_aware_specialist_plan
from mempool.routing_dataset import read_routing_records


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected task list in {path}")
    return data


def read_screened_task_ids(path: Path) -> set[str]:
    data = read_json(path)
    by_task = data.get("by_task", [])
    if isinstance(by_task, dict):
        task_ids = {str(task_id) for task_id in by_task}
    else:
        task_ids = {str(item["task_id"]) for item in by_task}
    task_ids.update(str(item["task_id"]) for item in data.get("records", []) if "task_id" in item)
    task_ids.update(str(task_id) for task_id in data.get("universal_failure_task_ids", []))
    task_ids.update(str(task_id) for task_id in data.get("candidate_task_ids", []))
    return task_ids


def load_task_union(paths: list[Path]) -> list[dict]:
    by_id = {}
    for path in paths:
        for task in read_tasks(path):
            by_id.setdefault(str(task["id"]), task)
    return [by_id[task_id] for task_id in sorted(by_id)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan fresh specialist acquisition candidates with a solvability prior."
    )
    parser.add_argument("--task-source", type=Path, action="append", required=True)
    parser.add_argument("--routing-dataset", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--target-worker", action="append", required=True)
    parser.add_argument("--exclude-screening-summary", type=Path, action="append", default=[])
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--per-worker-limit", type=int, default=4)
    parser.add_argument("--min-solvability-score", type=float)
    parser.add_argument("--tasks-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_task_union(args.task_source)
    routing_records = read_routing_records(args.routing_dataset)
    exclude_task_ids = set(args.exclude_task_id)
    for path in args.exclude_screening_summary:
        exclude_task_ids.update(read_screened_task_ids(path))
    plan = build_solvability_aware_specialist_plan(
        task_sources=tasks,
        routing_records=routing_records,
        candidate_report=read_json(args.candidate_report),
        target_workers=args.target_worker,
        exclude_task_ids=exclude_task_ids,
        per_worker_limit=args.per_worker_limit,
        min_solvability_score=args.min_solvability_score,
    )
    args.tasks_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.tasks_output.write_text(
        json.dumps(plan["selected_tasks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {key: value for key, value in plan.items() if key != "selected_tasks"}
    manifest["task_sources"] = [str(path) for path in args.task_source]
    manifest["routing_dataset"] = str(args.routing_dataset)
    manifest["candidate_report"] = str(args.candidate_report)
    manifest["exclude_screening_summaries"] = [str(path) for path in args.exclude_screening_summary]
    manifest["tasks_output"] = str(args.tasks_output)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
