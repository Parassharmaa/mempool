from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mempool.smoke_benchmark import (
    SmokeCodeBenchmarkAdapter,
    SmokeCodeTask,
    extract_python_code,
)


def load_task_lookup(task_paths: list[Path]) -> dict[str, tuple[SmokeCodeTask, Path]]:
    lookup: dict[str, tuple[SmokeCodeTask, Path]] = {}
    for path in task_paths:
        adapter = SmokeCodeBenchmarkAdapter(path)
        for task in adapter.load_tasks():
            lookup.setdefault(task.id, (task, path))
    return lookup


def load_executions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    executions = payload.get("executions", [])
    if not isinstance(executions, list):
        raise ValueError("comparison artifact missing executions list")
    return [dict(item) for item in executions]


def missing_eval_dependencies(result: dict[str, Any]) -> list[str]:
    metadata = result.get("metadata") or {}
    stderr = str(metadata.get("stderr_tail") or "")
    dependencies = sorted(set(re.findall(r"ModuleNotFoundError: No module named '([^']+)'", stderr)))
    return dependencies


def evaluate_execution(
    execution: dict[str, Any],
    *,
    task_lookup: dict[str, tuple[SmokeCodeTask, Path]],
    timeout_seconds: int,
) -> dict[str, Any]:
    task_id = str(execution.get("task_id"))
    row = {
        "schema_version": "mempool.orchestrated_execution_evaluation.v1",
        "task_id": task_id,
        "policy_id": execution.get("policy_id"),
        "selected_worker_id": (execution.get("selected_worker") or {}).get("id"),
        "selected_model": (execution.get("selected_worker") or {}).get("model"),
        "latency_ms": execution.get("latency_ms"),
        "response_chars": len(str((execution.get("response") or {}).get("content") or "")),
    }
    if task_id not in task_lookup:
        return {
            **row,
            "passed": False,
            "score": 0.0,
            "failure_mode": "missing_task",
            "task_file": None,
            "extracted_code_chars": 0,
            "result": {},
        }

    task, task_path = task_lookup[task_id]
    content = str((execution.get("response") or {}).get("content") or "")
    code = extract_python_code(content)
    adapter = SmokeCodeBenchmarkAdapter(task_path, timeout_seconds=timeout_seconds)
    result = adapter.evaluate_output(task, code)
    result_payload = asdict(result)
    missing_dependencies = missing_eval_dependencies(result_payload)
    failure_mode = "missing_eval_dependency" if missing_dependencies else result.failure_mode
    return {
        **row,
        "passed": result.passed,
        "score": result.score,
        "failure_mode": failure_mode,
        "task_file": str(task_path),
        "extracted_code_chars": len(code),
        "missing_eval_dependencies": missing_dependencies,
        "result": result_payload,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row.get("policy_id"))].append(row)
    policy_summaries = []
    for policy_id, policy_rows in sorted(by_policy.items()):
        passed = sum(1 for row in policy_rows if row["passed"])
        missing_dependency_count = sum(1 for row in policy_rows if row.get("failure_mode") == "missing_eval_dependency")
        evaluable_rows = [row for row in policy_rows if row.get("failure_mode") != "missing_eval_dependency"]
        evaluable_passed = sum(1 for row in evaluable_rows if row["passed"])
        missing_dependencies = sorted(
            {
                dependency
                for row in policy_rows
                for dependency in row.get("missing_eval_dependencies", [])
                if isinstance(dependency, str)
            }
        )
        policy_summaries.append(
            {
                "policy_id": policy_id,
                "record_count": len(policy_rows),
                "passed": passed,
                "pass_rate": passed / len(policy_rows) if policy_rows else 0.0,
                "missing_eval_dependency_count": missing_dependency_count,
                "missing_eval_dependencies": missing_dependencies,
                "evaluable_record_count": len(evaluable_rows),
                "evaluable_passed": evaluable_passed,
                "evaluable_pass_rate": evaluable_passed / len(evaluable_rows) if evaluable_rows else 0.0,
                "mean_latency_ms": sum(float(row.get("latency_ms") or 0.0) for row in policy_rows) / len(policy_rows)
                if policy_rows
                else 0.0,
            }
        )
    return {
        "record_count": len(rows),
        "missing_eval_dependency_count": sum(1 for row in rows if row.get("failure_mode") == "missing_eval_dependency"),
        "policy_summaries": policy_summaries,
    }


def evaluate_prompt_set(
    *,
    comparison_path: Path,
    task_paths: list[Path],
    output_path: Path,
    report_path: Path,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    lookup = load_task_lookup(task_paths)
    executions = load_executions(comparison_path)
    rows = [
        evaluate_execution(
            execution,
            task_lookup=lookup,
            timeout_seconds=timeout_seconds,
        )
        for execution in executions
    ]
    report = {
        "schema_version": "mempool.orchestrated_prompt_set_evaluation.v2",
        "comparison": str(comparison_path),
        "task_files": [str(path) for path in task_paths],
        "timeout_seconds": timeout_seconds,
        "task_lookup_count": len(lookup),
        **summarize(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate live orchestrated prompt-set executions against materialized SmokeCode/BigCodeBench tasks."
    )
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    args = parser.parse_args()
    report = evaluate_prompt_set(
        comparison_path=args.comparison,
        task_paths=args.task_file,
        output_path=args.output,
        report_path=args.report_output,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
