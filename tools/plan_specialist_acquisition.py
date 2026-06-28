from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.acquisition_plan import build_specialist_acquisition_plan
from mempool.routing_dataset import read_routing_records


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected task list in {path}")
    return data


def load_task_union(paths: list[Path]) -> list[dict]:
    by_id = {}
    for path in paths:
        for task in read_tasks(path):
            by_id.setdefault(str(task["id"]), task)
    return [by_id[task_id] for task_id in sorted(by_id)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan specialist BigCodeBench acquisition around current router misses."
    )
    parser.add_argument("--task-source", type=Path, action="append", required=True)
    parser.add_argument("--routing-dataset", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--target-worker", action="append", required=True)
    parser.add_argument("--comparison-worker", action="append", default=[])
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--per-worker-limit", type=int, default=4)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--tasks-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_task_union(args.task_source)
    routing_records = read_routing_records(args.routing_dataset)
    plan = build_specialist_acquisition_plan(
        task_sources=tasks,
        routing_records=routing_records,
        candidate_report=read_json(args.candidate_report),
        target_workers=args.target_worker,
        comparison_workers=args.comparison_worker,
        exclude_task_ids=set(args.exclude_task_id),
        per_worker_limit=args.per_worker_limit,
        repeat_count=args.repeat_count,
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
    manifest["tasks_output"] = str(args.tasks_output)
    manifest["run_command_template"] = (
        "PYTHONPATH=src python3 tools/run_real_smoke_benchmark.py "
        "--config <specialist-worker-pool.json> "
        f"--tasks {args.tasks_output} "
        f"--repeat-count {args.repeat_count} "
        "--eval-timeout-seconds 20 "
        "--run-id 20260628-specialist-acquisition "
        "--output research/evals/results/20260628-specialist-acquisition.json "
        "--outcomes research/evals/results/20260628-specialist-acquisition.jsonl "
        "--resume --progress"
    )
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
