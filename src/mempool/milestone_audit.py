from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EvidenceMap = dict[str, str]


MILESTONE_ORDER = [
    "M1-real-worker-pool-evaluation",
    "M2-outcome-dataset",
    "M3-lightweight-router",
    "M4-external-smoke-benchmark",
    "M5-small-trainable-orchestrator",
    "M5.5-agentic-harness-pilot",
    "M6-adaptive-memory-refresh",
]


DEFAULT_EVIDENCE: EvidenceMap = {
    "worker_outcomes": "research/evals/bigcodebench_hard_top4_offset0_outcomes.jsonl",
    "routing_dataset": "research/datasets/20260627-mixed-winner-23task-heldout-hard-routing.jsonl",
    "active_policy": "research/policies/active_policy.json",
    "external_smoke_report": "docs/benchmark_strategy.md",
    "external_comparison_report": "research/evals/results/20260627-selected-gated-fallback-active-23task.json",
    "m5_model": "research/models/20260628-m5-offline-multihead-50task.json",
    "m5_report": "research/evals/results/20260628-m5-offline-multihead-50task-report.json",
    "m5_gate": "research/evals/results/20260628-m5-offline-multihead-50task-policy-gate.json",
    "terminal_bench_report": "research/evals/terminal_bench_2p1_fix_git_oracle_vs_qwen_next_report.json",
    "terminal_bench_trajectories": "research/evals/terminal_bench_2p1_fix_git_qwen_next_trajectories.jsonl",
    "adaptive_refresh": "research/refreshes/20260628-router-miss-repeat-24task-profiled-refresh-cycle.json",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _exists(root: Path, evidence: EvidenceMap, key: str) -> bool:
    return _resolve(root, evidence[key]).exists()


def _artifact(root: Path, evidence: EvidenceMap, key: str) -> dict[str, Any]:
    path = _resolve(root, evidence[key])
    return {"key": key, "path": evidence[key], "exists": path.exists()}


def _routing_dataset_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    rows = read_jsonl(path)
    target_counts: dict[str, int] = {}
    workers_per_task = []
    solvable = 0
    for row in rows:
        target = str(row.get("target_worker_id") or "")
        target_counts[target] = target_counts.get(target, 0) + 1
        workers = row.get("workers") or []
        workers_per_task.append(len(workers))
        if any(bool(worker.get("passed")) for worker in workers):
            solvable += 1
    return {
        "exists": True,
        "task_count": len(rows),
        "target_worker_count": len([key for key in target_counts if key]),
        "target_counts": target_counts,
        "min_workers_per_task": min(workers_per_task) if workers_per_task else 0,
        "solvable_task_count": solvable,
    }


def _active_policy_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    active = read_json(path).get("active") or {}
    return {
        "exists": True,
        "model": active.get("model"),
        "dataset": active.get("dataset"),
        "loo": active.get("loo") or {},
        "target_mix": active.get("target_mix") or {},
    }


def _m5_summary(report_path: Path, gate_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"report_exists": report_path.exists(), "gate_exists": gate_path.exists()}
    if report_path.exists():
        report = read_json(report_path)
        evaluation = report.get("evaluation") or {}
        summary["evaluation"] = {
            "task_count": evaluation.get("task_count"),
            "target_accuracy": evaluation.get("target_accuracy"),
            "pass_at_1": evaluation.get("pass_at_1"),
            "solvable_pass_at_1": evaluation.get("solvable_pass_at_1"),
            "mean_latency_regret_ms": evaluation.get("mean_latency_regret_ms"),
            "workflow_accuracy": evaluation.get("workflow_accuracy"),
        }
        loo = report.get("leave_one_out") or {}
        if loo:
            summary["leave_one_out"] = {
                "task_count": loo.get("task_count"),
                "target_accuracy": loo.get("target_accuracy"),
                "pass_at_1": loo.get("pass_at_1"),
                "solvable_pass_at_1": loo.get("solvable_pass_at_1"),
                "mean_latency_regret_ms": loo.get("mean_latency_regret_ms"),
                "workflow_accuracy": loo.get("workflow_accuracy"),
            }
    if gate_path.exists():
        gate = read_json(gate_path)
        summary["gate_decision"] = gate.get("decision")
        summary["gate_reasons"] = gate.get("reasons") or []
    return summary


def _adaptive_refresh_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    refresh = read_json(path)
    guardrails = refresh.get("guardrails") or []
    return {
        "exists": True,
        "decision": refresh.get("decision") or refresh.get("gate_decision"),
        "promotion_profile": refresh.get("promotion_profile") or (refresh.get("gate") or {}).get("promotion_profile"),
        "guardrails_passed": all(bool(item.get("passed")) for item in guardrails),
        "guardrail_count": len(guardrails),
        "reasons": refresh.get("reasons") or [],
        "promotion_allowed": (refresh.get("promotion") or {}).get("allowed"),
    }


def _status(passed: list[bool], partial_when_any: bool = True) -> str:
    if all(passed):
        return "completed"
    if partial_when_any and any(passed):
        return "partial"
    return "pending"


def audit_milestones(root: Path = Path("."), evidence: EvidenceMap | None = None) -> dict[str, Any]:
    root = root.resolve()
    evidence = {**DEFAULT_EVIDENCE, **(evidence or {})}

    routing_path = _resolve(root, evidence["routing_dataset"])
    active_policy_path = _resolve(root, evidence["active_policy"])
    m5_report_path = _resolve(root, evidence["m5_report"])
    m5_gate_path = _resolve(root, evidence["m5_gate"])
    refresh_path = _resolve(root, evidence["adaptive_refresh"])

    routing = _routing_dataset_summary(routing_path)
    active = _active_policy_summary(active_policy_path)
    m5 = _m5_summary(m5_report_path, m5_gate_path)
    refresh = _adaptive_refresh_summary(refresh_path)

    milestones = []

    m1_passed = [
        _exists(root, evidence, "worker_outcomes") or _exists(root, evidence, "external_comparison_report"),
        bool(routing.get("min_workers_per_task", 0) >= 3),
        _exists(root, evidence, "external_smoke_report"),
    ]
    milestones.append(
        {
            "id": "M1-real-worker-pool-evaluation",
            "audit_status": _status(m1_passed),
            "summary": "Real worker outcomes and cost/latency-aware comparison artifacts exist.",
            "evidence": [
                _artifact(root, evidence, "worker_outcomes"),
                _artifact(root, evidence, "external_comparison_report"),
                _artifact(root, evidence, "external_smoke_report"),
            ],
            "metrics": {"routing_dataset": routing},
            "open_gaps": [] if all(m1_passed) else ["keep raw worker-outcome JSONL discoverable"],
        }
    )

    m2_passed = [
        routing.get("exists", False),
        bool(routing.get("task_count", 0) >= 1),
        bool(routing.get("target_worker_count", 0) >= 1),
    ]
    milestones.append(
        {
            "id": "M2-outcome-dataset",
            "audit_status": _status(m2_passed),
            "summary": "Canonical routing dataset exists with worker outcomes and routing targets.",
            "evidence": [_artifact(root, evidence, "routing_dataset")],
            "metrics": routing,
            "open_gaps": [] if all(m2_passed) else ["regenerate canonical routing records"],
        }
    )

    active_loo = active.get("loo") or {}
    m3_passed = [
        active.get("exists", False),
        bool(active.get("model")),
        bool(active.get("dataset")),
        bool(active_loo.get("available")),
    ]
    milestones.append(
        {
            "id": "M3-lightweight-router",
            "audit_status": _status(m3_passed),
            "summary": "Active lightweight logits router is trained, evaluated, and promoted through the registry.",
            "evidence": [_artifact(root, evidence, "active_policy")],
            "metrics": active,
            "open_gaps": [] if all(m3_passed) else ["restore active policy registry and LOO metrics"],
        }
    )

    m4_passed = [
        _exists(root, evidence, "external_smoke_report"),
        _exists(root, evidence, "external_comparison_report"),
        bool(active_loo.get("task_count", 0) >= 10),
    ]
    milestones.append(
        {
            "id": "M4-external-smoke-benchmark",
            "audit_status": _status(m4_passed),
            "summary": "External BigCodeBench smoke/pilot reports exist and compare learned routing with baselines.",
            "evidence": [
                _artifact(root, evidence, "external_smoke_report"),
                _artifact(root, evidence, "external_comparison_report"),
            ],
            "metrics": {
                "active_loo_task_count": active_loo.get("task_count"),
                "active_solvable_pass_at_1": active_loo.get("solvable_pass_at_1"),
            },
            "open_gaps": [] if all(m4_passed) else ["write a single compact M4 benchmark summary artifact"],
        }
    )

    m5_passed = [
        _exists(root, evidence, "m5_model"),
        m5.get("report_exists", False),
        m5.get("gate_exists", False),
    ]
    m5_open_gaps = []
    if m5.get("gate_decision") == "quarantine":
        m5_open_gaps.append("candidate is quarantined; active promoted policy remains the 23-task lightweight router")
    if not all(m5_passed):
        m5_open_gaps.append("complete local multi-head orchestrator artifact, report, and gate")
    milestones.append(
        {
            "id": "M5-small-trainable-orchestrator",
            "audit_status": _status(m5_passed),
            "summary": "Local multi-head trainable orchestrator exists with a clear negative/quarantine result.",
            "evidence": [
                _artifact(root, evidence, "m5_model"),
                _artifact(root, evidence, "m5_report"),
                _artifact(root, evidence, "m5_gate"),
            ],
            "metrics": m5,
            "open_gaps": m5_open_gaps,
        }
    )

    m55_passed = [
        _exists(root, evidence, "terminal_bench_report"),
        _exists(root, evidence, "terminal_bench_trajectories"),
    ]
    milestones.append(
        {
            "id": "M5.5-agentic-harness-pilot",
            "audit_status": _status(m55_passed),
            "summary": "Terminal-Bench pilot evidence exists, but this remains a side track until BigCodeBench routing improves.",
            "evidence": [
                _artifact(root, evidence, "terminal_bench_report"),
                _artifact(root, evidence, "terminal_bench_trajectories"),
            ],
            "metrics": {},
            "open_gaps": [] if all(m55_passed) else ["finish reproducible Terminal-Bench subset report and trajectory ledger"],
        }
    )

    m6_passed = [
        refresh.get("exists", False),
        bool(refresh.get("guardrails_passed")),
        refresh.get("decision") in {"promote", "quarantine"},
    ]
    m6_open_gaps = []
    if refresh.get("decision") == "quarantine":
        profile = refresh.get("promotion_profile")
        if profile:
            m6_open_gaps.append(f"refresh cycle works but produced a quarantined candidate under {profile} gate")
        else:
            m6_open_gaps.append("refresh cycle works but produced a quarantined candidate")
    if not all(m6_passed):
        m6_open_gaps.append("complete evaluated refresh cycle with guardrails")
    milestones.append(
        {
            "id": "M6-adaptive-memory-refresh",
            "audit_status": _status(m6_passed),
            "summary": "Adaptive refresh path is implemented end to end with privacy, rollback, and promotion gates.",
            "evidence": [_artifact(root, evidence, "adaptive_refresh")],
            "metrics": refresh,
            "open_gaps": m6_open_gaps,
        }
    )

    recommended_active = next(
        (
            milestone["id"]
            for milestone in milestones
            if milestone["audit_status"] != "completed"
        ),
        milestones[-1]["id"],
    )
    return {
        "schema_version": "mempool.milestone_audit.v1",
        "root": str(root),
        "milestones": milestones,
        "recommended_active_milestone": recommended_active,
        "recommendations": [
            "Keep the 23-task lightweight router as the active promoted policy.",
            "Use the quarantined 50-task multi-head result as the next training-signal target, not as a deployable policy.",
            "Use preserve_accuracy for promotion-grade adaptive refreshes unless an experiment explicitly studies a latency-for-accuracy tradeoff.",
            "Next research loop should improve stable specialist wins and broad-pass latency rows before another promotion attempt.",
        ],
    }


def apply_milestone_audit(milestones_path: Path, audit: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(milestones_path)
    status_by_id = {
        milestone["id"]: milestone["audit_status"]
        for milestone in audit.get("milestones", [])
    }
    for milestone in payload.get("milestones", []):
        audit_status = status_by_id.get(milestone.get("id"))
        if audit_status == "completed":
            milestone["status"] = "completed"
        elif audit_status == "partial":
            milestone["status"] = "active"
        elif audit_status == "pending":
            milestone["status"] = "pending"
    payload["active_milestone"] = audit.get("recommended_active_milestone")
    payload["last_audit"] = {
        "schema_version": audit.get("schema_version"),
        "recommended_active_milestone": audit.get("recommended_active_milestone"),
    }
    return payload
