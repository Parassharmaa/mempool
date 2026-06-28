from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.acquisition_plan import build_catalog_candidate_acquisition_plan


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected task list in {path}")
    return data


def load_task_union(paths: list[Path]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for path in paths:
        for task in read_tasks(path):
            by_id.setdefault(str(task["id"]), task)
    return [by_id[task_id] for task_id in sorted(by_id)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan a bounded BigCodeBench run for newly cataloged cloud workers."
    )
    parser.add_argument("--current-pool", type=Path, required=True)
    parser.add_argument("--task-source", type=Path, action="append", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--measured-worker-id", action="append", default=[])
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--tasks-output", type=Path, required=True)
    parser.add_argument("--pool-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    current_pool = read_json(args.current_pool)
    tasks = load_task_union(args.task_source)
    plan = build_catalog_candidate_acquisition_plan(
        current_pool=current_pool,
        task_sources=tasks,
        task_ids=args.task_id,
        measured_worker_ids=set(args.measured_worker_id),
        repeat_count=args.repeat_count,
    )

    args.tasks_output.parent.mkdir(parents=True, exist_ok=True)
    args.pool_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.tasks_output.write_text(
        json.dumps(plan["selected_tasks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.pool_output.write_text(
        json.dumps(plan["worker_pool"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        key: value
        for key, value in plan.items()
        if key not in {"selected_tasks", "worker_pool"}
    }
    manifest["tasks_output"] = str(args.tasks_output)
    manifest["pool_output"] = str(args.pool_output)
    manifest["run_command"] = (
        "PYTHONPATH=src python3 tools/run_real_smoke_benchmark.py "
        f"--config {args.pool_output} "
        f"--tasks {args.tasks_output} "
        f"--repeat-count {args.repeat_count} "
        "--eval-timeout-seconds 20 "
        "--run-id 20260627-catalog-candidate-regression-slices "
        "--output research/evals/results/20260627-catalog-candidate-regression-slices.json "
        "--outcomes research/evals/results/20260627-catalog-candidate-regression-slices.jsonl "
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
