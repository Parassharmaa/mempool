from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from mempool.smoke_benchmark import SmokeCodeBenchmarkAdapter, SmokeCodeTask, task_to_dict


IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+(.+))")


def imported_roots(source: str) -> list[str]:
    roots: set[str] = set()
    for line in source.splitlines():
        match = IMPORT_RE.match(line)
        if not match:
            continue
        from_module, import_modules = match.groups()
        if from_module:
            roots.add(from_module.split(".", 1)[0])
            continue
        for module in import_modules.split(","):
            name = module.strip().split(" as ", 1)[0].strip()
            if name:
                roots.add(name.split(".", 1)[0])
    return sorted(roots)


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def task_import_roots(task: SmokeCodeTask) -> list[str]:
    sources = [task.prompt, *task.tests]
    roots: set[str] = set()
    for source in sources:
        roots.update(imported_roots(source))
    return sorted(roots)


def load_unique_tasks(task_paths: list[Path]) -> list[tuple[SmokeCodeTask, Path]]:
    tasks: list[tuple[SmokeCodeTask, Path]] = []
    seen: set[str] = set()
    for path in task_paths:
        adapter = SmokeCodeBenchmarkAdapter(path)
        for task in adapter.load_tasks():
            if task.id in seen:
                continue
            seen.add(task.id)
            tasks.append((task, path))
    return tasks


def filter_evaluable_tasks(
    *,
    task_paths: list[Path],
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    rows = []
    kept: list[SmokeCodeTask] = []
    for task, source_path in load_unique_tasks(task_paths):
        imports = task_import_roots(task)
        missing = [module for module in imports if not module_available(module)]
        evaluable = not missing
        if evaluable:
            kept.append(task)
        rows.append(
            {
                "task_id": task.id,
                "source_file": str(source_path),
                "import_roots": imports,
                "missing_import_roots": missing,
                "evaluable_in_current_env": evaluable,
            }
        )

    report = {
        "schema_version": "mempool.evaluable_task_filter.v1",
        "task_files": [str(path) for path in task_paths],
        "input_task_count": len(rows),
        "evaluable_task_count": len(kept),
        "excluded_task_count": len(rows) - len(kept),
        "missing_import_roots": sorted({module for row in rows for module in row["missing_import_roots"]}),
        "tasks": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([task_to_dict(task) for task in kept], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter materialized SmokeCode/BigCodeBench tasks to those whose imports are available locally."
    )
    parser.add_argument("--task-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    report = filter_evaluable_tasks(
        task_paths=args.task_file,
        output_path=args.output,
        report_path=args.report_output,
    )
    print(
        json.dumps(
            {
                "input_task_count": report["input_task_count"],
                "evaluable_task_count": report["evaluable_task_count"],
                "excluded_task_count": report["excluded_task_count"],
                "missing_import_roots": report["missing_import_roots"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
