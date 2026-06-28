from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any


CONTENT_FIELDS = {
    "instruction",
    "instructions",
    "prompt",
    "description",
    "solution",
    "oracle",
    "tests",
    "test_script",
}

DEFAULT_CATEGORIES = (
    "software-engineering",
    "system-administration",
    "data-processing",
    "security",
    "model-training",
)

DIFFICULTY_RANK = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _categories(row: dict[str, Any]) -> set[str]:
    values = _as_list(row.get("categories")) + _as_list(row.get("category"))
    return {value.lower() for value in values if value}


def _difficulty(row: dict[str, Any]) -> str:
    value = str(row.get("difficulty", "medium")).lower()
    if value not in DIFFICULTY_RANK:
        return "medium"
    return value


def validate_metadata_only(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        forbidden = sorted(CONTENT_FIELDS.intersection(row))
        if forbidden:
            task_id = row.get("id", "<unknown>")
            joined = ", ".join(forbidden)
            raise ValueError(
                f"Terminal-Bench pilot selection only accepts metadata; "
                f"{task_id} includes forbidden content fields: {joined}"
            )
        if not row.get("id"):
            raise ValueError("Terminal-Bench candidate is missing id")


def select_terminal_bench_pilot(
    rows: list[dict[str, Any]],
    limit: int,
    excluded_ids: set[str] | None = None,
    preferred_categories: tuple[str, ...] = DEFAULT_CATEGORIES,
) -> dict[str, Any]:
    validate_metadata_only(rows)
    excluded_ids = excluded_ids or set()
    candidates = [row for row in rows if str(row["id"]) not in excluded_ids]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        return (-DIFFICULTY_RANK[_difficulty(row)], str(row["id"]))

    for category in preferred_categories:
        if len(selected) >= limit:
            break
        matching = [
            row
            for row in candidates
            if str(row["id"]) not in selected_ids and category in _categories(row)
        ]
        if not matching:
            continue
        picked = min(matching, key=sort_key)
        selected.append({**picked, "selection_reason": f"category coverage: {category}"})
        selected_ids.add(str(picked["id"]))

    while len(selected) < limit:
        remaining = [
            row for row in candidates if str(row["id"]) not in selected_ids
        ]
        if not remaining:
            break
        picked = min(remaining, key=sort_key)
        selected.append({**picked, "selection_reason": "difficulty-balanced filler"})
        selected_ids.add(str(picked["id"]))

    return {
        "benchmark_id": "terminal-bench-2.1",
        "mode": "metadata-only-pilot-selection",
        "candidate_count": len(candidates),
        "excluded_count": len(excluded_ids),
        "target_task_count": limit,
        "selected_task_ids": [str(row["id"]) for row in selected],
        "selected": selected,
        "run_contract": {
            "task_content_policy": "Do not persist benchmark task instructions, oracle solutions, or verifier code in mempool training corpora.",
            "first_run": "Compare single-worker agent runs against the active orchestrator-selected worker policy.",
            "record_fields": [
                "task_id",
                "worker_id",
                "policy_id",
                "terminal_actions",
                "file_edits",
                "test_result",
                "latency_ms",
                "cost_usd",
                "failure_mode",
            ],
        },
    }


def _strip_content(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in CONTENT_FIELDS}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_terminal_bench_metadata(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.is_file():
            data = _read_json(path)
            items = data if isinstance(data, list) else [data]
            rows.extend(_strip_content(dict(item)) for item in items)
            continue
        for task_file in sorted(path.rglob("task.toml")):
            data = tomllib.loads(task_file.read_text(encoding="utf-8"))
            task = data.get("task", {})
            metadata = data.get("metadata", {})
            rows.append(
                {
                    "id": str(task.get("name") or task_file.parent.relative_to(path)),
                    "category": metadata.get("category", "software-engineering"),
                    "difficulty": metadata.get("difficulty", "medium"),
                    "tags": list(metadata.get("tags", [])),
                }
            )
        for task_file in sorted(path.rglob("task.yaml")):
            rel = task_file.parent.relative_to(path)
            parts = rel.parts
            category = parts[1] if len(parts) > 2 and parts[0] == "tasks" else (parts[0] if parts else "software-engineering")
            rows.append(
                {
                    "id": str(rel),
                    "categories": [category],
                    "difficulty": "medium",
                }
            )
    return rows


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def summarize_harbor_job(job_dir: Path) -> dict[str, Any]:
    result = _load_optional_json(job_dir / "result.json")
    config = _load_optional_json(job_dir / "config.json")
    stats = result.get("stats", {})
    completed = int(stats.get("n_completed_trials", 0) or 0)
    errored = int(stats.get("n_errored_trials", 0) or 0)
    cancelled = int(stats.get("n_cancelled_trials", 0) or 0)
    running = int(stats.get("n_running_trials", 0) or 0)
    pending = int(stats.get("n_pending_trials", 0) or 0)
    finished = result.get("finished_at")
    if finished and errored == 0 and cancelled == 0 and running == 0 and pending == 0:
        status = "complete"
    elif running or pending:
        status = "running_or_stale"
    elif not finished and (completed or errored or cancelled):
        status = "interrupted_ambiguous"
    else:
        status = "missing_or_unknown"
    trial_dirs = [path for path in job_dir.iterdir()] if job_dir.exists() else []
    trial_count = sum(1 for path in trial_dirs if path.is_dir())
    return {
        "job_dir": str(job_dir),
        "job_id": result.get("id"),
        "status": status,
        "environment_type": (config.get("environment") or {}).get("type"),
        "agent_type": (config.get("agent") or {}).get("type"),
        "trial_directory_count": trial_count,
        "n_total_trials": result.get("n_total_trials"),
        "n_completed_trials": completed,
        "n_errored_trials": errored,
        "n_cancelled_trials": cancelled,
        "n_running_trials": running,
        "n_pending_trials": pending,
        "cost_usd": stats.get("cost_usd"),
        "raw_log_policy": "not_read",
    }


TRAJECTORY_FORBIDDEN_FIELDS = {"stdout", "stderr", "raw_log", "command", "content"}


def validate_terminal_bench_trajectories(records: list[dict[str, Any]]) -> list[str]:
    errors = []
    for index, record in enumerate(records):
        for field in CONTENT_FIELDS & set(record):
            errors.append(f"record {index} includes forbidden content fields: {field}")
        for collection_name in ("terminal_actions", "tests_run"):
            for item in record.get(collection_name, []):
                forbidden = sorted(TRAJECTORY_FORBIDDEN_FIELDS & set(item))
                if forbidden:
                    errors.append(f"record {index} {collection_name} includes forbidden content fields: {', '.join(forbidden)}")
    return errors


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def harbor_job_to_terminal_bench_trajectories(
    job_dir: Path,
    run_id: str,
    agent_id: str,
    worker_id: str,
    policy_id: str,
) -> list[dict[str, Any]]:
    records = []
    for result_path in sorted(job_dir.glob("*/result.json")):
        data = _load_optional_json(result_path)
        task_id_value = data.get("task_id")
        if isinstance(task_id_value, dict):
            task_id = Path(str(task_id_value.get("path", ""))).name
        else:
            task_id = Path(str(task_id_value or result_path.parent.name)).name
        started = _parse_time(data.get("started_at"))
        finished = _parse_time(data.get("finished_at"))
        latency_ms = (finished - started).total_seconds() * 1000 if started and finished else None
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0.0)
        success = float(reward or 0.0) > 0.0
        records.append(
            {
                "benchmark_id": "terminal-bench-2.1",
                "run_id": run_id,
                "task_id": task_id,
                "trial_id": str(data.get("id") or result_path.parent.name),
                "agent_id": agent_id,
                "worker_id": worker_id,
                "policy_id": policy_id,
                "selected_workflow": "terminal-agent",
                "task_success": success,
                "verifier_passed": success,
                "latency_ms": latency_ms,
                "cost_usd": (data.get("agent_result") or {}).get("cost_usd"),
                "terminal_actions": [],
                "file_edits": [],
                "tests_run": [{"command_summary": "verifier", "passed": success}],
                "worker_switches": [],
                "failure_mode": None if success else "verifier_failed",
                "raw_log_policy": "not_read",
            }
        )
    return records


def compare_terminal_bench_trajectories(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_policy.setdefault(str(record["policy_id"]), []).append(record)
        by_task.setdefault(str(record["task_id"]), []).append(record)
    summaries = []
    for policy_id, rows in sorted(by_policy.items()):
        successes = sum(1 for row in rows if row.get("task_success"))
        summaries.append(
            {
                "policy_id": policy_id,
                "record_count": len(rows),
                "success_rate": successes / len(rows) if rows else 0.0,
                "mean_latency_ms": sum(float(row.get("latency_ms") or 0.0) for row in rows) / len(rows) if rows else 0.0,
            }
        )
    return {
        "record_count": len(records),
        "task_count": len(by_task),
        "policy_summaries": summaries,
        "task_comparisons": [
            {
                "task_id": task_id,
                "policies": {str(row["policy_id"]): bool(row.get("task_success")) for row in rows},
            }
            for task_id, rows in sorted(by_task.items())
        ],
    }


def evaluate_terminal_bench_readiness(summary_paths: list[Path]) -> dict[str, Any]:
    checks = []
    for path in summary_paths:
        data = _read_json(path)
        reasons = []
        process_status = data.get("process_status")
        harbor_status = (data.get("harbor_summary") or {}).get("status")
        if process_status not in {None, "exited"}:
            reasons.append(f"process_status={process_status}")
        if harbor_status != "complete":
            reasons.append(f"harbor_status={harbor_status}")
        checks.append({"path": str(path), "ready": not reasons, "reasons": reasons})
    return {"ready": all(check["ready"] for check in checks), "checks": checks}


def refresh_terminal_bench_preflight_summary(summary_path: Path) -> dict[str, Any]:
    legacy = _read_json(summary_path)
    job_dir = Path(legacy["job_dir"])
    refreshed = {
        **legacy,
        "process_status": legacy.get("process_status"),
        "harbor_summary": summarize_harbor_job(job_dir),
        "raw_log_policy": "not_read",
        "refresh_policy": "metadata_only_result_json_and_config_json",
    }
    summary_path.write_text(json.dumps(refreshed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return refreshed
