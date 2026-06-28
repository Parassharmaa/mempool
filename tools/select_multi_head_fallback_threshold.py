from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.multi_head_orchestrator import (
    evaluate_multi_head_fallback_predictions,
    read_substrate,
    validate_substrate_records,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    evaluation = candidate["evaluation"]
    return (
        float(evaluation["solvable_pass_at_1"]),
        float(evaluation["pass_at_1"]),
        float(evaluation["target_accuracy"]),
        -float(evaluation["mean_latency_regret_ms"]),
        -float(evaluation["mean_latency_ms"]),
        -float(evaluation["fallback_rate"]),
    )


def regret_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    evaluation = candidate["evaluation"]
    return (
        float(evaluation["mean_latency_regret_ms"]),
        -float(evaluation["solvable_pass_at_1"]),
        -float(evaluation["pass_at_1"]),
        -float(evaluation["target_accuracy"]),
    )


def select_multi_head_fallback_threshold(
    *,
    substrate_path: Path,
    report_path: Path,
    margins: list[float],
    verifier_thresholds: list[float],
    max_attempts: int = 2,
) -> dict[str, Any]:
    if not margins:
        raise ValueError("at least one margin is required")
    if not verifier_thresholds:
        raise ValueError("at least one verifier threshold is required")

    records = read_substrate(substrate_path)
    errors = validate_substrate_records(records)
    if errors:
        return {"valid": False, "errors": errors}

    report = read_json(report_path)
    loo = report.get("leave_one_out") or {}
    predictions = loo.get("predictions") or []
    if not predictions:
        return {
            "valid": False,
            "errors": ["report has no leave-one-out predictions"],
            "report": str(report_path),
        }
    missing_distribution = [
        prediction.get("task_id")
        for prediction in predictions
        if "worker_distribution" not in prediction
    ]
    if missing_distribution:
        return {
            "valid": False,
            "errors": ["leave-one-out predictions are missing worker_distribution"],
            "missing_task_ids": missing_distribution[:10],
            "report": str(report_path),
        }

    candidates = []
    for margin in margins:
        for threshold in verifier_thresholds:
            evaluation = evaluate_multi_head_fallback_predictions(
                records,
                predictions,
                max_attempts=max_attempts,
                max_first_second_margin=margin,
                min_verifier_probability=threshold,
            )
            candidates.append(
                {
                    "max_attempts": max_attempts,
                    "max_first_second_margin": margin,
                    "min_verifier_probability": threshold,
                    "evaluation": evaluation,
                }
            )

    selected = max(candidates, key=rank_key)
    base_pass = float(loo.get("pass_at_1", 0.0) or 0.0)
    pass_gain_candidates = [
        candidate
        for candidate in candidates
        if float(candidate["evaluation"]["pass_at_1"]) > base_pass
    ]
    lowest_regret_pass_gain = (
        min(pass_gain_candidates, key=regret_key)
        if pass_gain_candidates
        else None
    )
    return {
        "valid": True,
        "substrate": str(substrate_path),
        "report": str(report_path),
        "base_model": report.get("model_output"),
        "base_leave_one_out": {
            key: loo.get(key)
            for key in [
                "target_accuracy",
                "pass_at_1",
                "solvable_pass_at_1",
                "mean_latency_regret_ms",
                "mean_latency_ms",
            ]
        },
        "selection_rule": [
            "maximize solvable_pass_at_1",
            "maximize pass_at_1",
            "maximize target_accuracy",
            "minimize mean_latency_regret_ms",
            "minimize mean_latency_ms",
            "minimize fallback_rate",
        ],
        "selected": selected,
        "lowest_regret_pass_gain": lowest_regret_pass_gain,
        "candidates": candidates,
    }


def policy_payload(selection: dict[str, Any]) -> dict[str, Any]:
    if not selection.get("valid"):
        raise ValueError("cannot build policy payload for invalid selection")
    selected = selection["selected"]
    return {
        "policy": "multi-head-gated-fallback",
        "base_model": selection["base_model"],
        "substrate": selection["substrate"],
        "report": selection["report"],
        "max_attempts": selected["max_attempts"],
        "max_first_second_margin": selected["max_first_second_margin"],
        "min_verifier_probability": selected["min_verifier_probability"],
        "evaluation": selected["evaluation"],
        "selection_rule": selection["selection_rule"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a gated fallback threshold for a multi-head orchestrator report."
    )
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--verifier-thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
    args = parser.parse_args()

    selection = select_multi_head_fallback_threshold(
        substrate_path=args.substrate,
        report_path=args.report,
        margins=args.margins,
        verifier_thresholds=args.verifier_thresholds,
        max_attempts=args.max_attempts,
    )
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not selection.get("valid"):
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 1

    policy = policy_payload(selection)
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selection": selection, "policy": policy}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
