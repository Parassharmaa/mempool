from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_task_ids_from_task_file(path: Path) -> set[str]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"expected task list in {path}")
    return {str(task["id"]) for task in data}


def read_task_ids_from_fallback_report(path: Path) -> set[str]:
    data = read_json(path)
    return {str(item["task_id"]) for item in data.get("selected", [])}


def select_screen_batch(
    acquisition_report: dict[str, Any],
    task_source: list[dict[str, Any]],
    limit: int,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    exclude_ids = exclude_ids or set()
    tasks_by_id = {str(task["id"]): task for task in task_source}
    ranked = [
        item
        for item in acquisition_report.get("ranked_candidates", [])
        if item["task_id"] in tasks_by_id and item["task_id"] not in exclude_ids
    ]
    selected = ranked[:limit]
    return {
        "source_report": acquisition_report.get("task_sources", []),
        "candidate_count": len(ranked),
        "excluded_count": len(exclude_ids),
        "selected_task_ids": [item["task_id"] for item in selected],
        "selected_tasks": [tasks_by_id[item["task_id"]] for item in selected],
        "selected": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a cheap one-sample solvability screen batch from fallback candidates."
    )
    parser.add_argument("--acquisition-report", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--exclude-task-file", type=Path, action="append", default=[])
    parser.add_argument("--exclude-fallback-report", type=Path, action="append", default=[])
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    exclude_ids = set(args.exclude_task_id)
    for path in args.exclude_task_file:
        exclude_ids.update(read_task_ids_from_task_file(path))
    for path in args.exclude_fallback_report:
        exclude_ids.update(read_task_ids_from_fallback_report(path))

    selection = select_screen_batch(
        read_json(args.acquisition_report),
        read_json(args.tasks),
        limit=args.limit,
        exclude_ids=exclude_ids,
    )
    report = {
        "acquisition_report": str(args.acquisition_report),
        "tasks": str(args.tasks),
        "excluded_task_files": [str(path) for path in args.exclude_task_file],
        "excluded_fallback_reports": [str(path) for path in args.exclude_fallback_report],
        "excluded_task_ids": sorted(exclude_ids),
        **{key: value for key, value in selection.items() if key != "selected_tasks"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection["selected_tasks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
