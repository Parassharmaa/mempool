from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mempool.logits_router import evaluate_logits_router
from mempool.router_baseline import (
    FamilyRouter,
    NearestNeighborRouter,
    evaluate_policy,
    fastest_worker,
    leave_one_out_predictions,
    strongest_worker,
)
from mempool.routing_dataset import read_routing_records, validate_routing_records
try:
    from tools.evaluate_active_policy import load_active_router
    from tools.evaluate_latency_calibrated_logits_router import (
        conditionally_calibrated_predictions,
        probe_gated_calibrate_task_ids,
        read_json,
    )
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router
    from evaluate_latency_calibrated_logits_router import (
        conditionally_calibrated_predictions,
        probe_gated_calibrate_task_ids,
        read_json,
    )
from mempool.latency_calibrated_router import evaluate_latency_calibrated_predictions
from mempool.logits_router import LogitsRouter


def summarize(evaluation: object) -> dict[str, object]:
    data = asdict(evaluation)
    data["target_accuracy"] = evaluation.target_accuracy
    data["pass_at_1"] = evaluation.pass_at_1
    data["solvable_pass_at_1"] = evaluation.solvable_pass_at_1
    data["solvable_target_accuracy"] = evaluation.solvable_target_accuracy
    data["mean_latency_ms"] = evaluation.mean_latency_ms
    data["mean_target_latency_ms"] = evaluation.mean_target_latency_ms
    data["mean_latency_regret_ms"] = evaluation.mean_latency_regret_ms
    data["cost_per_solved_task"] = evaluation.cost_per_solved_task
    return data


def evaluate_probe_gated_policy(records: list[dict], policy_path: Path) -> tuple[dict, dict]:
    policy = read_json(policy_path)
    if policy.get("policy") != "probe-gated-latency-calibrated-logits-router":
        raise ValueError(f"unsupported probe-gated policy: {policy.get('policy')}")

    router = LogitsRouter.from_dict(read_json(Path(policy["base_model"]))["router"])
    probe_gate = policy["probe_gate"]
    calibration = policy["calibration"]
    calibrate_task_ids = probe_gated_calibrate_task_ids(
        records=records,
        probe_worker_ids=list(probe_gate["probe_worker_ids"]),
        probe_mode=str(probe_gate.get("mode", "all")),
        min_pass_rate=float(probe_gate.get("min_pass_rate", 1.0)),
    )
    predictions = conditionally_calibrated_predictions(
        records=records,
        router=router,
        calibrate_task_ids=calibrate_task_ids,
        latency_cost_per_second=float(calibration["latency_cost_per_second"]),
        min_probability_ratio=float(calibration.get("min_probability_ratio", 0.0)),
        min_probability=float(calibration.get("min_probability", 0.0)),
    )
    evaluation = evaluate_latency_calibrated_predictions(
        records,
        predictions,
        latency_cost_per_second=0.0,
        min_probability_ratio=0.0,
        min_probability=0.0,
    )
    evaluation["policy"] = "probe-gated-latency-calibrated-logits-router"
    evaluation["probe_gate"] = {
        **probe_gate,
        "calibrated_task_count": len(calibrate_task_ids),
        "calibrated_task_ids": sorted(calibrate_task_ids),
    }
    evaluation["calibration"] = calibration
    return evaluation, policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate lightweight router baseline.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-policy-registry", type=Path)
    parser.add_argument("--probe-gated-policy", type=Path)
    args = parser.parse_args()

    records = read_routing_records(args.dataset)
    errors = validate_routing_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    router = FamilyRouter.train(records)
    nearest_router = NearestNeighborRouter.train(records)
    family_predictions = [router.predict(record) for record in records]
    nearest_predictions = [nearest_router.predict(record) for record in records]
    strongest = strongest_worker(records)
    fastest = fastest_worker(records)
    strongest_predictions = [strongest for _ in records]
    fastest_predictions = [fastest for _ in records]
    oracle_predictions = [record["target_worker_id"] for record in records]
    evaluations = [
        summarize(evaluate_policy(records, "family-router", family_predictions)),
        summarize(evaluate_policy(records, "nearest-neighbor-router", nearest_predictions)),
    ]
    if len(records) > 1:
        loo_family_predictions = leave_one_out_predictions(records, "family")
        loo_nearest_predictions = leave_one_out_predictions(records, "nearest-neighbor")
        evaluations.extend(
            [
                summarize(evaluate_policy(records, "family-router-loo", loo_family_predictions)),
                summarize(evaluate_policy(records, "nearest-neighbor-router-loo", loo_nearest_predictions)),
            ]
        )
    else:
        evaluations.extend(
            [
                {
                    "policy": "family-router-loo",
                    "task_count": len(records),
                    "available": False,
                    "reason": "leave-one-out requires at least two records",
                },
                {
                    "policy": "nearest-neighbor-router-loo",
                    "task_count": len(records),
                    "available": False,
                    "reason": "leave-one-out requires at least two records",
                },
            ]
        )
    evaluations.extend(
        [
            summarize(evaluate_policy(records, "strongest-worker", strongest_predictions)),
            summarize(evaluate_policy(records, "fastest-worker", fastest_predictions)),
            summarize(evaluate_policy(records, "oracle-target", oracle_predictions)),
        ]
    )
    active_policy = None
    if args.active_policy_registry:
        active_router, active = load_active_router(args.active_policy_registry)
        active_evaluation = evaluate_logits_router(records, active_router)
        active_evaluation["policy"] = "active-logits-router"
        evaluations.append(active_evaluation)
        active_policy = {
            "registry": str(args.active_policy_registry),
            "model": active["model"],
            "dataset": active["dataset"],
        }

    probe_gated_policy = None
    if args.probe_gated_policy:
        probe_gated_evaluation, probe_gated_payload = evaluate_probe_gated_policy(
            records,
            args.probe_gated_policy,
        )
        evaluations.append(probe_gated_evaluation)
        probe_gated_policy = {
            "artifact": str(args.probe_gated_policy),
            "base_model": probe_gated_payload["base_model"],
            "probe_gate": probe_gated_payload["probe_gate"],
            "calibration": probe_gated_payload["calibration"],
        }

    payload = {
        "dataset": str(args.dataset),
        "router": router.to_dict(),
        "nearest_router": nearest_router.to_dict(),
        "baselines": {
            "strongest_worker": strongest,
            "fastest_worker": fastest,
        },
        "evaluations": evaluations,
    }
    if active_policy:
        payload["active_policy"] = active_policy
    if probe_gated_policy:
        payload["probe_gated_policy"] = probe_gated_policy
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
