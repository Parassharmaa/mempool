from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .task_features import feature_safe_name
from .terminal_bench import validate_terminal_bench_trajectories


SCHEMA_VERSION = "mempool.agentic_turn_substrate.v1"

ACTION_KINDS = (
    "shell",
    "edit",
    "test",
    "inspect",
    "reason",
    "other",
)

WORKFLOW_KINDS = (
    "terminal-agent",
    "direct",
    "verify_then_fallback",
)


def _one_hot(label: str, labels: list[str]) -> dict[str, float]:
    return {candidate: 1.0 if candidate == label else 0.0 for candidate in labels}


def _action_kind(action: dict[str, Any]) -> str:
    value = feature_safe_name(str(action.get("action_kind", "other")))
    if value in ACTION_KINDS:
        return value
    return "other"


def _workflow_kind(record: dict[str, Any]) -> str:
    value = str(record.get("selected_workflow") or "terminal-agent")
    if value in WORKFLOW_KINDS:
        return value
    return "terminal-agent"


def _worker_ids(records: list[dict[str, Any]]) -> list[str]:
    ids = {
        str(record.get("worker_id"))
        for record in records
        if record.get("worker_id")
    }
    for record in records:
        for switch in record.get("worker_switches", []):
            if switch.get("from_worker_id"):
                ids.add(str(switch["from_worker_id"]))
            if switch.get("to_worker_id"):
                ids.add(str(switch["to_worker_id"]))
    return sorted(ids)


def _state_features(record: dict[str, Any], action: dict[str, Any], turn_index: int) -> dict[str, float]:
    terminal_actions = record.get("terminal_actions", [])
    tests_run = record.get("tests_run", [])
    file_edits = record.get("file_edits", [])
    exit_code = action.get("exit_code")
    failed_exit = exit_code is not None and int(exit_code) != 0
    tests_failed = any(not bool(test.get("passed")) for test in tests_run)
    return {
        "bias": 1.0,
        "turn_index": float(turn_index),
        "turn_fraction": float(turn_index / max(1, len(terminal_actions) - 1)),
        "terminal_action_count": float(len(terminal_actions)),
        "file_edit_count": float(len(file_edits)),
        "test_count": float(len(tests_run)),
        "worker_switch_count": float(len(record.get("worker_switches", []))),
        "failed_exit_seen": float(failed_exit),
        "test_failure_seen": float(tests_failed),
        "task_success": float(bool(record.get("task_success"))),
        "verifier_passed": float(bool(record.get("verifier_passed"))),
        "latency_seconds": float(record.get("latency_ms") or 0.0) / 1000.0,
        "cost_microusd": float(record.get("cost_usd") or 0.0) * 1_000_000.0,
    }


def _target(
    record: dict[str, Any],
    action: dict[str, Any],
    *,
    turn_index: int,
    worker_ids: list[str],
) -> dict[str, Any]:
    action_kind = _action_kind(action)
    workflow_kind = _workflow_kind(record)
    worker_id = str(action.get("worker_id") or record.get("worker_id"))
    is_last_turn = turn_index == max(0, len(record.get("terminal_actions", [])) - 1)
    exit_code = action.get("exit_code")
    failed_exit = exit_code is not None and int(exit_code) != 0
    tests_failed = any(not bool(test.get("passed")) for test in record.get("tests_run", []))
    should_repair = (failed_exit or tests_failed) and not bool(record.get("task_success"))
    should_stop = is_last_turn and bool(record.get("task_success"))
    return {
        "worker_distribution": _one_hot(worker_id, worker_ids),
        "target_worker_id": worker_id,
        "workflow_kind": workflow_kind,
        "workflow_distribution": _one_hot(workflow_kind, list(WORKFLOW_KINDS)),
        "action_kind": action_kind,
        "action_distribution": _one_hot(action_kind, list(ACTION_KINDS)),
        "verifier_probability": 1.0 if action_kind == "test" or is_last_turn else 0.0,
        "repair_probability": 1.0 if should_repair else 0.0,
        "stop_probability": 1.0 if should_stop else 0.0,
        "memory_update_probability": 1.0 if is_last_turn and bool(record.get("task_success")) else 0.0,
    }


def build_agentic_turn_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = validate_terminal_bench_trajectories(records)
    if errors:
        raise ValueError(f"invalid terminal trajectories: {errors}")

    worker_ids = _worker_ids(records)
    examples = []
    for record in records:
        actions = record.get("terminal_actions") or [
            {
                "index": 0,
                "action_kind": "reason",
                "summary": "no terminal action captured",
                "exit_code": None,
            }
        ]
        for turn_index, action in enumerate(actions):
            target = _target(record, action, turn_index=turn_index, worker_ids=worker_ids)
            state_features = _state_features(record, action, turn_index)
            state_summary = {
                "benchmark_id": record.get("benchmark_id"),
                "run_id": record.get("run_id"),
                "task_id": record.get("task_id"),
                "trial_id": record.get("trial_id"),
                "agent_id": record.get("agent_id"),
                "policy_id": record.get("policy_id"),
                "turn_index": turn_index,
                "action_summary": action.get("summary"),
                "previous_action_kind": _action_kind(actions[turn_index - 1]) if turn_index else None,
                "failure_mode": record.get("failure_mode"),
            }
            examples.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark_id": record.get("benchmark_id"),
                    "run_id": record.get("run_id"),
                    "task_id": record.get("task_id"),
                    "trial_id": record.get("trial_id"),
                    "turn_id": f"{record.get('trial_id')}:{turn_index}",
                    "turn_index": turn_index,
                    "state_summary": state_summary,
                    "dense_features": state_features,
                    "available_workers": worker_ids,
                    "target": target,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a compact turn-level orchestration model. "
                                "Predict worker, workflow, action, verifier, repair, stop, and memory-update heads as JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(state_summary, sort_keys=True),
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(target, sort_keys=True),
                        },
                    ],
                }
            )
    return examples


def build_agentic_turn_substrate(
    *,
    trajectory_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples = build_agentic_turn_examples(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples),
        encoding="utf-8",
    )

    target_workers: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for example in examples:
        target = example["target"]
        target_worker = str(target["target_worker_id"])
        action_kind = str(target["action_kind"])
        target_workers[target_worker] = target_workers.get(target_worker, 0) + 1
        action_counts[action_kind] = action_counts.get(action_kind, 0) + 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_path": str(trajectory_path),
        "output": str(output_path),
        "record_count": len(examples),
        "trajectory_count": len(records),
        "worker_ids": _worker_ids(records),
        "target_worker_counts": target_workers,
        "action_counts": action_counts,
        "training_status": "schema_ready_not_trained",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
