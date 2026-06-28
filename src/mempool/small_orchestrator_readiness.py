from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .routing_dataset import read_routing_records, validate_routing_records
from .orchestrator_contract import validate_multi_head_contract


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_exists(path_value: str | None) -> bool:
    return bool(path_value) and Path(path_value).exists()


def _metric(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _active_policy_check(
    registry_path: Path,
    *,
    min_tasks: int,
    min_target_workers: int,
    min_loo_target_accuracy: float,
    min_loo_solvable_pass_at_1: float,
    max_loo_latency_regret_ms: float,
) -> dict[str, Any]:
    if not registry_path.exists():
        return {
            "name": "active_policy",
            "passed": False,
            "reasons": [f"missing active policy registry: {registry_path}"],
        }

    registry = read_json(registry_path)
    active = registry.get("active") or {}
    model_path = active.get("model")
    dataset_path = active.get("dataset")
    loo = active.get("loo") or {}
    target_mix = active.get("target_mix") or {}
    reasons = []

    if not _path_exists(model_path):
        reasons.append(f"missing active model artifact: {model_path}")
    if not _path_exists(dataset_path):
        reasons.append(f"missing active dataset artifact: {dataset_path}")
    if not loo.get("available"):
        reasons.append("active policy does not include leave-one-out metrics")

    task_count = int(loo.get("task_count") or target_mix.get("task_count") or 0)
    target_worker_count = int(target_mix.get("target_worker_count") or 0)
    loo_target_accuracy = _metric(loo.get("target_accuracy"))
    loo_solvable_pass = _metric(loo.get("solvable_pass_at_1"))
    loo_latency_regret = _metric(loo.get("mean_latency_regret_ms"))

    if task_count < min_tasks:
        reasons.append(f"active dataset has {task_count} tasks; need at least {min_tasks}")
    if target_worker_count < min_target_workers:
        reasons.append(
            f"active dataset has {target_worker_count} target workers; "
            f"need at least {min_target_workers}"
        )
    if loo.get("available") and loo_target_accuracy < min_loo_target_accuracy:
        reasons.append(
            f"LOO target accuracy {loo_target_accuracy:.3f} "
            f"is below {min_loo_target_accuracy:.3f}"
        )
    if loo.get("available") and loo_solvable_pass < min_loo_solvable_pass_at_1:
        reasons.append(
            f"LOO solvable pass@1 {loo_solvable_pass:.3f} "
            f"is below {min_loo_solvable_pass_at_1:.3f}"
        )
    if loo.get("available") and loo_latency_regret > max_loo_latency_regret_ms:
        reasons.append(
            f"LOO latency regret {loo_latency_regret:.1f} ms "
            f"exceeds {max_loo_latency_regret_ms:.1f} ms"
        )

    return {
        "name": "active_policy",
        "passed": not reasons,
        "reasons": reasons,
        "evidence": {
            "registry": str(registry_path),
            "model": model_path,
            "dataset": dataset_path,
            "task_count": task_count,
            "target_worker_count": target_worker_count,
            "loo_target_accuracy": loo_target_accuracy,
            "loo_solvable_pass_at_1": loo_solvable_pass,
            "loo_latency_regret_ms": loo_latency_regret,
        },
    }


def _dataset_check(dataset_path: Path | None, *, min_workers_per_task: int) -> dict[str, Any]:
    if dataset_path is None or not dataset_path.exists():
        return {
            "name": "routing_dataset",
            "passed": False,
            "reasons": [f"missing routing dataset: {dataset_path}"],
        }

    records = read_routing_records(dataset_path)
    validation_errors = validate_routing_records(records)
    worker_counts = [len(record.get("workers", [])) for record in records]
    min_observed_workers = min(worker_counts) if worker_counts else 0
    target_counts: dict[str, int] = {}
    solvable_count = 0
    for record in records:
        target = str(record.get("target_worker_id", ""))
        target_counts[target] = target_counts.get(target, 0) + 1
        if any(bool(worker.get("passed")) for worker in record.get("workers", [])):
            solvable_count += 1

    reasons = list(validation_errors)
    if min_observed_workers < min_workers_per_task:
        reasons.append(
            f"minimum workers per task is {min_observed_workers}; "
            f"need at least {min_workers_per_task}"
        )

    return {
        "name": "routing_dataset",
        "passed": not reasons,
        "reasons": reasons,
        "evidence": {
            "dataset": str(dataset_path),
            "task_count": len(records),
            "solvable_task_count": solvable_count,
            "min_workers_per_task": min_observed_workers,
            "target_counts": target_counts,
        },
    }


def _model_shape_check(model_path: Path | None) -> dict[str, Any]:
    if model_path is None or not model_path.exists():
        return {
            "name": "worker_logits_head",
            "passed": False,
            "reasons": [f"missing model artifact: {model_path}"],
        }

    payload = read_json(model_path)
    router = payload.get("router") or {}
    model_type = str(payload.get("model_type", ""))
    reasons = []
    if "logits" not in model_type and router.get("policy") != "logits-router":
        reasons.append(f"model is not a logits router: {model_type or router.get('policy')}")
    if not router.get("worker_ids"):
        reasons.append("router has no worker ids")
    if not router.get("feature_names"):
        reasons.append("router has no feature names")
    if not router.get("weights"):
        reasons.append("router has no weights")

    return {
        "name": "worker_logits_head",
        "passed": not reasons,
        "reasons": reasons,
        "evidence": {
            "model": str(model_path),
            "model_type": model_type,
            "worker_count": len(router.get("worker_ids") or []),
            "feature_count": len(router.get("feature_names") or []),
        },
    }


def _report_check(
    name: str,
    report_path: Path,
    *,
    minimum_solvable_pass_at_1: float | None = None,
    require_passed_flag: bool = False,
) -> dict[str, Any]:
    if not report_path.exists():
        return {
            "name": name,
            "passed": False,
            "reasons": [f"missing report: {report_path}"],
        }

    payload = read_json(report_path)
    evaluation = payload.get("evaluation") or payload
    reasons = []
    if require_passed_flag and not bool(payload.get("passed")):
        reasons.append("report passed flag is false")
    if minimum_solvable_pass_at_1 is not None:
        solvable_pass = _metric(evaluation.get("solvable_pass_at_1"))
        if solvable_pass < minimum_solvable_pass_at_1:
            reasons.append(
                f"solvable pass@1 {solvable_pass:.3f} "
                f"is below {minimum_solvable_pass_at_1:.3f}"
            )

    return {
        "name": name,
        "passed": not reasons,
        "reasons": reasons,
        "evidence": {
            "report": str(report_path),
            "policy": evaluation.get("policy") or payload.get("policy"),
            "task_count": evaluation.get("task_count"),
            "solvable_pass_at_1": evaluation.get("solvable_pass_at_1"),
            "target_accuracy": evaluation.get("target_accuracy"),
            "mean_latency_regret_ms": evaluation.get("mean_latency_regret_ms"),
        },
    }


def _action_space_check(
    *,
    worker_logits_ready: bool,
    fallback_report_ready: bool,
    require_workflow_head: bool,
    require_abstain_head: bool,
    orchestrator_contract_path: Path | None = None,
) -> dict[str, Any]:
    reasons = []
    contract_valid = False
    contract_heads: dict[str, Any] = {}
    if orchestrator_contract_path and orchestrator_contract_path.exists():
        contract = read_json(orchestrator_contract_path)
        contract_heads = contract.get("heads") or {}
        contract_valid = bool(validate_multi_head_contract(contract).get("valid"))
    if not worker_logits_ready:
        reasons.append("worker-distribution logits head is not ready")
    if not fallback_report_ready:
        reasons.append("conditional verifier/fallback signal is not ready")
    if require_workflow_head and "workflow_kind" not in contract_heads:
        reasons.append("workflow-kind logits head artifact is not implemented")
    if require_abstain_head and "abstain_probability" not in contract_heads:
        reasons.append("abstain probability head artifact is not implemented")
    if orchestrator_contract_path and not contract_valid:
        reasons.append("orchestrator contract is invalid")

    return {
        "name": "m5_action_space",
        "passed": not reasons,
        "reasons": reasons,
        "evidence": {
            "worker_distribution_logits": worker_logits_ready,
            "conditional_verifier_or_fallback_signal": fallback_report_ready,
            "workflow_kind_logits": "workflow_kind" in contract_heads,
            "abstain_probability": "abstain_probability" in contract_heads,
            "orchestrator_contract": str(orchestrator_contract_path) if orchestrator_contract_path else None,
        },
    }


def audit_small_orchestrator_readiness(
    *,
    registry_path: Path = Path("research/policies/active_policy.json"),
    fallback_report_path: Path = Path("research/evals/results/20260627-selected-gated-fallback-active-23task.json"),
    regression_report_path: Path = Path("research/evals/results/20260627-selected-gated-fallback-regression-slices.json"),
    orchestrator_contract_path: Path | None = None,
    min_tasks: int = 50,
    min_target_workers: int = 4,
    min_workers_per_task: int = 4,
    min_loo_target_accuracy: float = 0.75,
    min_loo_solvable_pass_at_1: float = 0.80,
    max_loo_latency_regret_ms: float = 750.0,
    min_fallback_solvable_pass_at_1: float = 0.90,
    require_workflow_head: bool = True,
    require_abstain_head: bool = True,
) -> dict[str, Any]:
    active_check = _active_policy_check(
        registry_path,
        min_tasks=min_tasks,
        min_target_workers=min_target_workers,
        min_loo_target_accuracy=min_loo_target_accuracy,
        min_loo_solvable_pass_at_1=min_loo_solvable_pass_at_1,
        max_loo_latency_regret_ms=max_loo_latency_regret_ms,
    )
    active_evidence = active_check.get("evidence") or {}
    dataset_path = Path(active_evidence["dataset"]) if active_evidence.get("dataset") else None
    model_path = Path(active_evidence["model"]) if active_evidence.get("model") else None

    dataset_check = _dataset_check(dataset_path, min_workers_per_task=min_workers_per_task)
    model_check = _model_shape_check(model_path)
    fallback_check = _report_check(
        "conditional_verifier_or_fallback",
        fallback_report_path,
        minimum_solvable_pass_at_1=min_fallback_solvable_pass_at_1,
    )
    regression_check = _report_check(
        "fallback_regression_slices",
        regression_report_path,
        require_passed_flag=True,
    )
    action_space_check = _action_space_check(
        worker_logits_ready=model_check["passed"],
        fallback_report_ready=fallback_check["passed"] and regression_check["passed"],
        require_workflow_head=require_workflow_head,
        require_abstain_head=require_abstain_head,
        orchestrator_contract_path=orchestrator_contract_path,
    )

    checks = [
        active_check,
        dataset_check,
        model_check,
        fallback_check,
        regression_check,
        action_space_check,
    ]
    ready = all(check["passed"] for check in checks)
    reasons = [
        f"{check['name']}: {reason}"
        for check in checks
        for reason in check.get("reasons", [])
    ]
    next_actions = []
    if active_evidence.get("task_count", 0) < min_tasks:
        next_actions.append("collect more repeated BigCodeBench routing rows before backbone training")
    if not action_space_check["passed"]:
        next_actions.append("add explicit workflow-kind and abstain heads to the orchestrator output contract")
    if fallback_check["passed"] and regression_check["passed"]:
        next_actions.append("treat the current gated fallback as the verifier-probability teacher signal")
    if ready:
        next_actions.append("start the smallest M5 experiment: compact non-autoregressive multi-head logits model")

    return {
        "ready_for_m5_small_orchestrator": ready,
        "decision": "start-m5" if ready else "continue-m3-data-and-heads",
        "checks": checks,
        "reasons": reasons,
        "thresholds": {
            "min_tasks": min_tasks,
            "min_target_workers": min_target_workers,
            "min_workers_per_task": min_workers_per_task,
            "min_loo_target_accuracy": min_loo_target_accuracy,
            "min_loo_solvable_pass_at_1": min_loo_solvable_pass_at_1,
            "max_loo_latency_regret_ms": max_loo_latency_regret_ms,
            "min_fallback_solvable_pass_at_1": min_fallback_solvable_pass_at_1,
            "require_workflow_head": require_workflow_head,
            "require_abstain_head": require_abstain_head,
        },
        "next_actions": next_actions,
    }
