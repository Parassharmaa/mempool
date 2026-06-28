from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.latency_calibrated_router import evaluate_latency_calibrated_predictions
from mempool.logits_router import LogitsRouter, evaluate_logits_router
from mempool.multi_head_orchestrator import (
    MultiHeadOrchestrator,
    read_substrate,
    validate_substrate_records,
)
from mempool.routing_dataset import read_routing_records, validate_routing_records

LOGITS_ROUTER_MODEL_TYPES = {"logits-router", "linear-softmax-logits-router"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def concise_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: evaluation.get(key)
        for key in [
            "policy",
            "task_count",
            "target_accuracy",
            "pass_at_1",
            "solvable_pass_at_1",
            "mean_latency_ms",
            "mean_target_latency_ms",
            "mean_latency_regret_ms",
            "cost_per_solved_task",
            "mean_kl",
            "changed_from_top",
            "change_rate",
        ]
        if key in evaluation
    }


def load_active_router(registry_path: Path) -> tuple[LogitsRouter, dict[str, Any]]:
    registry = read_json(registry_path)
    active = registry.get("active")
    if not active:
        raise ValueError(f"no active policy in {registry_path}")
    model_path = Path(active["model"])
    payload = read_json(model_path)
    if payload.get("model_type") and payload.get("model_type") not in LOGITS_ROUTER_MODEL_TYPES:
        raise ValueError(f"active policy is not a logits router: {payload.get('model_type')}")
    return LogitsRouter.from_dict(payload["router"]), active


def load_active_policy(registry_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = read_json(registry_path)
    active = registry.get("active")
    if not active:
        raise ValueError(f"no active policy in {registry_path}")
    model_path = Path(active["model"])
    payload = read_json(model_path)
    return payload, active


def evaluate_logits_active_policy(
    payload: dict[str, Any],
    active: dict[str, Any],
    dataset_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    dataset = dataset_path or Path(active["dataset"])
    records = read_routing_records(dataset)
    errors = validate_routing_records(records)
    if errors:
        return {"valid": False, "errors": errors}, dataset
    router = LogitsRouter.from_dict(payload["router"])
    return evaluate_logits_router(records, router), dataset


def _load_multi_head_model(payload: dict[str, Any]) -> MultiHeadOrchestrator:
    base_model = read_json(Path(payload["base_model"]))
    return MultiHeadOrchestrator.from_dict(base_model["orchestrator"])


def evaluate_latency_calibrated_active_policy(
    payload: dict[str, Any],
    dataset_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    dataset = dataset_path or Path(payload["substrate"])
    records = read_substrate(dataset)
    errors = validate_substrate_records(records)
    if errors:
        return {"valid": False, "errors": errors}, dataset
    model = _load_multi_head_model(payload)
    predictions = [model.predict(record) for record in records]
    calibration = payload["calibration"]
    return (
        evaluate_latency_calibrated_predictions(
            records,
            predictions,
            latency_cost_per_second=float(calibration["latency_cost_per_second"]),
            min_probability_ratio=float(calibration["min_probability_ratio"]),
            min_probability=float(calibration["min_probability"]),
        ),
        dataset,
    )


def evaluate_active_policy_payload(
    payload: dict[str, Any],
    active: dict[str, Any],
    dataset_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    model_type = str(payload.get("model_type", "logits-router"))
    if model_type in LOGITS_ROUTER_MODEL_TYPES:
        return evaluate_logits_active_policy(payload, active, dataset_path)
    if model_type == "latency-calibrated-multi-head-router":
        return evaluate_latency_calibrated_active_policy(payload, dataset_path)
    raise ValueError(f"unsupported active policy model_type: {model_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the active logits-router policy.")
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--dataset", type=Path, help="Defaults to the active policy dataset.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload, active = load_active_policy(args.registry)
    evaluation, dataset = evaluate_active_policy_payload(payload, active, args.dataset)
    if evaluation.get("valid") is False:
        print(json.dumps(evaluation, indent=2))
        return 1
    payload = {
        "registry": str(args.registry),
        "policy_type": payload.get("model_type", "logits-router"),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "evaluated_dataset": str(dataset),
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "registry": payload["registry"],
                "policy_type": payload["policy_type"],
                "active_model": payload["active_model"],
                "evaluated_dataset": payload["evaluated_dataset"],
                "evaluation": concise_evaluation(evaluation),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
