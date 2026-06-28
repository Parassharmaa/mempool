from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.routing_dataset import read_routing_records, validate_routing_records
from mempool.second_attempt_value import (
    SecondAttemptValueHead,
    evaluate_learned_value_head,
    train_second_attempt_value_head,
)

try:
    from tools.evaluate_active_policy import load_active_router
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router


def router_predictions(records: list[dict[str, Any]], router: Any) -> list[dict[str, Any]]:
    predictions = []
    for record in records:
        distribution = router.distribution(record)
        top_probability = max(distribution.values()) if distribution else 0.0
        predictions.append(
            {
                "worker_distribution": distribution,
                "verifier_probability": 1.0 - top_probability,
                "abstain_probability": 0.0,
            }
        )
    return predictions


def evaluate_regression_slices(
    registry: Path,
    manifest: Path,
    head: SecondAttemptValueHead,
    latency_cost_per_second: float,
) -> dict[str, Any]:
    router, active = load_active_router(registry)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    results = []
    for item in payload.get("slices", []):
        dataset = Path(item["dataset"])
        records = read_routing_records(dataset)
        errors = validate_routing_records(records)
        if errors:
            results.append(
                {
                    "id": item["id"],
                    "dataset": str(dataset),
                    "passed": False,
                    "errors": errors,
                }
            )
            continue
        predictions = router_predictions(records, router)
        evaluation = evaluate_learned_value_head(
            records,
            predictions,
            head,
            latency_cost_per_second=latency_cost_per_second,
        )
        min_solvable = float(item.get("minimum_solvable_pass_at_1", 0.0))
        expected_count = item.get("expected_solvable_task_count")
        count_ok = (
            expected_count is None
            or int(evaluation["solvable_task_count"]) == int(expected_count)
        )
        metric_ok = float(evaluation["solvable_pass_at_1"]) >= min_solvable
        results.append(
            {
                "id": item["id"],
                "dataset": str(dataset),
                "note": item.get("note"),
                "passed": count_ok and metric_ok,
                "minimum_solvable_pass_at_1": min_solvable,
                "expected_solvable_task_count": expected_count,
                "evaluation": evaluation,
            }
        )
    return {
        "registry": str(registry),
        "manifest": str(manifest),
        "policy": "learned-second-attempt-value-head",
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "latency_cost_per_second": latency_cost_per_second,
        "threshold": head.threshold,
        "passed": all(item["passed"] for item in results),
        "results": results,
    }


def rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    evaluation = candidate["active_evaluation"]
    return (
        float(evaluation["solvable_pass_at_1"]),
        float(evaluation["pass_at_1"]),
        float(evaluation["target_accuracy"]),
        -float(evaluation["mean_latency_regret_ms"]),
        -float(evaluation["mean_latency_ms"]),
        -float(evaluation["fallback_rate"]),
    )


def select_policy(
    registry: Path,
    dataset: Path,
    regression_manifest: Path,
    latency_costs: list[float],
    thresholds: list[float],
    epochs: int,
    learning_rate: float,
    l2: float,
) -> dict[str, Any]:
    if not latency_costs:
        raise ValueError("at least one latency cost is required")
    if not thresholds:
        raise ValueError("at least one threshold is required")
    router, active = load_active_router(registry)
    records = read_routing_records(dataset)
    errors = validate_routing_records(records)
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "registry": str(registry),
            "dataset": str(dataset),
        }
    predictions = router_predictions(records, router)
    candidates = []
    trained_heads = {}
    histories = {}
    for latency_cost in latency_costs:
        head, history = train_second_attempt_value_head(
            records,
            predictions,
            latency_cost_per_second=latency_cost,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        trained_heads[latency_cost] = head
        histories[latency_cost] = history
        for threshold in thresholds:
            candidate_head = SecondAttemptValueHead(
                weights=list(head.weights),
                threshold=threshold,
            )
            active_evaluation = evaluate_learned_value_head(
                records,
                predictions,
                candidate_head,
                latency_cost_per_second=latency_cost,
            )
            regression_report = evaluate_regression_slices(
                registry,
                regression_manifest,
                candidate_head,
                latency_cost,
            )
            candidates.append(
                {
                    "latency_cost_per_second": latency_cost,
                    "threshold": threshold,
                    "eligible": bool(regression_report["passed"]),
                    "active_evaluation": active_evaluation,
                    "regression_report": regression_report,
                    "head": {
                        "weights": candidate_head.weights,
                        "threshold": candidate_head.threshold,
                    },
                    "training_history": history,
                }
            )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(eligible, key=rank_key) if eligible else None
    return {
        "valid": True,
        "registry": str(registry),
        "base_model": active["model"],
        "base_dataset": active["dataset"],
        "evaluated_dataset": str(dataset),
        "regression_manifest": str(regression_manifest),
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "latency_costs": latency_costs,
            "thresholds": thresholds,
        },
        "selection_rule": [
            "require all regression slices to pass",
            "maximize active solvable_pass_at_1",
            "maximize active pass_at_1",
            "maximize active target_accuracy",
            "minimize active mean_latency_regret_ms",
            "minimize active mean_latency_ms",
            "minimize active fallback_rate",
        ],
        "selected": selected,
        "candidates": candidates,
    }


def policy_payload(selection: dict[str, Any]) -> dict[str, Any]:
    selected = selection.get("selected")
    if not selected:
        raise ValueError("cannot write policy without a selected candidate")
    return {
        "policy": "learned-second-attempt-value-head",
        "base_model": selection["base_model"],
        "base_dataset": selection["base_dataset"],
        "evaluated_dataset": selection["evaluated_dataset"],
        "regression_manifest": selection["regression_manifest"],
        "latency_cost_per_second": selected["latency_cost_per_second"],
        "head": selected["head"],
        "training": {
            **selection["training"],
            "history": selected["training_history"],
        },
        "active_evaluation": selected["active_evaluation"],
        "regression_passed": selected["regression_report"]["passed"],
        "selection_rule": selection["selection_rule"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and select a learned second-attempt value policy."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--regression-manifest",
        type=Path,
        default=Path("research/evals/router_regression_slices.json"),
    )
    parser.add_argument("--latency-costs", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
    args = parser.parse_args()

    selection = select_policy(
        registry=args.registry,
        dataset=args.dataset,
        regression_manifest=args.regression_manifest,
        latency_costs=args.latency_costs,
        thresholds=args.thresholds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not selection.get("valid", False):
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 1
    if not selection.get("selected"):
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 2

    policy = policy_payload(selection)
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selection": selection, "policy": policy}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
