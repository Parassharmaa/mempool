from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.acquisition_source import build_execution_manifest


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize the recommended acquisition source as an executable manifest."
    )
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--selected-tasks", type=Path, required=True)
    parser.add_argument("--worker-pool", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--eval-timeout-seconds", type=int, default=20)
    parser.add_argument("--required-package", action="append", default=[])
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--tasks-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    args = parser.parse_args()

    selected_tasks = read_json(args.selected_tasks)
    if not isinstance(selected_tasks, list):
        raise ValueError("--selected-tasks must contain a task list")
    manifest = build_execution_manifest(
        source_plan=read_json(args.source_plan),
        source_report=read_json(args.source_report),
        selected_tasks=selected_tasks,
        worker_pool=read_json(args.worker_pool),
        worker_pool_path=args.worker_pool,
        tasks_output=args.tasks_output,
        run_id=args.run_id,
        output_path=args.output,
        outcomes_path=args.outcomes,
        repeat_count=args.repeat_count,
        eval_timeout_seconds=args.eval_timeout_seconds,
        required_packages=list(args.required_package),
        python_executable=args.python_executable,
    )

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
