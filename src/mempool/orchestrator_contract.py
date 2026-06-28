from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


WORKFLOW_KINDS = ["direct", "verify_then_fallback"]
REQUIRED_HEADS = {
    "worker_distribution",
    "workflow_kind",
    "verifier_probability",
    "abstain_probability",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def build_multi_head_contract(
    *,
    active_policy_registry: Path,
    fallback_report_path: Path,
    regression_report_path: Path,
    output_path: Path | None = None,
    schema_version: str = "mempool.orchestrator_contract.v1",
    direct_workflow_probability: float = 0.85,
    fallback_workflow_probability: float = 0.15,
    abstain_probability: float = 0.05,
) -> dict[str, Any]:
    registry = read_json(active_policy_registry)
    active = registry.get("active") or {}
    active_model_path = Path(active["model"])
    router_payload = read_json(active_model_path)
    router = router_payload.get("router") or {}
    fallback_report = read_json(fallback_report_path)
    regression_report = read_json(regression_report_path)
    fallback_eval = fallback_report.get("evaluation") or fallback_report
    fallback_rate = float(fallback_eval.get("fallback_rate", 0.0) or 0.0)

    direct_logit = logit(direct_workflow_probability)
    fallback_logit = logit(fallback_workflow_probability)
    abstain_logit = logit(abstain_probability)
    verifier_logit = logit(max(fallback_rate, abstain_probability))

    contract = {
        "schema_version": schema_version,
        "contract_type": "multi-head-orchestrator-logits-contract",
        "status": "bridge-contract",
        "purpose": (
            "Defines the M5 action-logit surface before training a compact "
            "neural or language-model backbone."
        ),
        "sources": {
            "active_policy_registry": str(active_policy_registry),
            "active_worker_model": str(active_model_path),
            "active_dataset": active.get("dataset"),
            "fallback_report": str(fallback_report_path),
            "regression_report": str(regression_report_path),
        },
        "heads": {
            "worker_distribution": {
                "head_type": "softmax",
                "source": "active_logits_router",
                "worker_ids": list(router.get("worker_ids") or []),
                "feature_names": list(router.get("feature_names") or []),
                "weights": router.get("weights") or [],
                "training_target": "soft worker reward distribution",
                "loss": "kl_divergence",
            },
            "workflow_kind": {
                "head_type": "softmax",
                "source": "calibrated_bridge_prior",
                "labels": WORKFLOW_KINDS,
                "logits": [direct_logit, fallback_logit],
                "training_target": "future terminal or fallback trajectory workflow labels",
                "loss": "cross_entropy",
            },
            "verifier_probability": {
                "head_type": "sigmoid",
                "source": "selected_gated_fallback_teacher",
                "logit": verifier_logit,
                "threshold": 0.5,
                "teacher_policy": fallback_eval.get("policy"),
                "teacher_fallback_rate": fallback_eval.get("fallback_rate"),
                "teacher_solvable_pass_at_1": fallback_eval.get("solvable_pass_at_1"),
                "regression_passed": bool(regression_report.get("passed")),
                "training_target": "fallback/verifier usefulness probability",
                "loss": "binary_cross_entropy",
            },
            "abstain_probability": {
                "head_type": "sigmoid",
                "source": "calibrated_bridge_prior",
                "logit": abstain_logit,
                "threshold": 0.5,
                "training_target": "future unsolved or high-risk task labels",
                "loss": "binary_cross_entropy",
            },
        },
        "promotion_requirements": {
            "min_repeated_tasks": 50,
            "requires_leave_one_out": True,
            "requires_terminal_bench_pilot_before_memory_refresh": True,
            "requires_backbone_latency_report": True,
        },
    }
    validation = validate_multi_head_contract(contract)
    contract["validation"] = validation
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def validate_multi_head_contract(contract: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    heads = contract.get("heads")
    if not isinstance(heads, dict):
        return {
            "valid": False,
            "reasons": ["heads must be an object"],
            "head_count": 0,
            "available_heads": [],
        }

    available_heads = set(heads)
    missing = sorted(REQUIRED_HEADS - available_heads)
    if missing:
        reasons.append(f"missing required heads: {missing}")

    worker_head = heads.get("worker_distribution") or {}
    if worker_head.get("head_type") != "softmax":
        reasons.append("worker_distribution must be a softmax head")
    if not worker_head.get("worker_ids"):
        reasons.append("worker_distribution has no worker ids")
    if not worker_head.get("weights"):
        reasons.append("worker_distribution has no weights")

    workflow_head = heads.get("workflow_kind") or {}
    if workflow_head.get("head_type") != "softmax":
        reasons.append("workflow_kind must be a softmax head")
    if workflow_head.get("labels") != WORKFLOW_KINDS:
        reasons.append(f"workflow_kind labels must be {WORKFLOW_KINDS}")
    if len(workflow_head.get("logits") or []) != len(WORKFLOW_KINDS):
        reasons.append("workflow_kind logits must match labels")

    for head_name in ["verifier_probability", "abstain_probability"]:
        head = heads.get(head_name) or {}
        if head.get("head_type") != "sigmoid":
            reasons.append(f"{head_name} must be a sigmoid head")
        if "logit" not in head:
            reasons.append(f"{head_name} is missing logit")
        if "threshold" not in head:
            reasons.append(f"{head_name} is missing threshold")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "head_count": len(available_heads),
        "available_heads": sorted(available_heads),
    }
