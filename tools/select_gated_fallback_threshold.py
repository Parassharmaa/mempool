from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.conditional_policy import evaluate_gated_fallback
from mempool.routing_dataset import read_routing_records, validate_routing_records

try:
    from tools.evaluate_active_policy import load_active_router
    from tools.evaluate_router_regression_slices import evaluate_slices
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router
    from evaluate_router_regression_slices import evaluate_slices


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


def select_threshold(
    registry: Path,
    dataset: Path | None,
    regression_manifest: Path,
    margins: list[float],
    max_attempts: int = 2,
) -> dict[str, Any]:
    if not margins:
        raise ValueError("at least one margin is required")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    router, active = load_active_router(registry)
    evaluated_dataset = dataset or Path(active["dataset"])
    records = read_routing_records(evaluated_dataset)
    errors = validate_routing_records(records)
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "registry": str(registry),
            "dataset": str(evaluated_dataset),
        }

    candidates = []
    for margin in margins:
        active_evaluation = evaluate_gated_fallback(
            records,
            router,
            max_attempts=max_attempts,
            max_first_second_margin=margin,
        )
        regression_report = evaluate_slices(
            registry,
            regression_manifest,
            policy="gated",
            max_attempts=max_attempts,
            max_first_second_margin=margin,
        )
        candidates.append(
            {
                "max_first_second_margin": margin,
                "max_attempts": max_attempts,
                "eligible": bool(regression_report["passed"]),
                "active_evaluation": active_evaluation,
                "regression_report": regression_report,
            }
        )

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(eligible, key=rank_key) if eligible else None
    return {
        "valid": True,
        "registry": str(registry),
        "base_model": active["model"],
        "base_dataset": active["dataset"],
        "evaluated_dataset": str(evaluated_dataset),
        "regression_manifest": str(regression_manifest),
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
        raise ValueError("cannot write policy payload without a selected candidate")
    return {
        "policy": "gated-fallback",
        "base_model": selection["base_model"],
        "base_dataset": selection["base_dataset"],
        "evaluated_dataset": selection["evaluated_dataset"],
        "regression_manifest": selection["regression_manifest"],
        "max_attempts": selected["max_attempts"],
        "max_first_second_margin": selected["max_first_second_margin"],
        "active_evaluation": selected["active_evaluation"],
        "regression_passed": selected["regression_report"]["passed"],
        "selection_rule": selection["selection_rule"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a gated fallback threshold for the active logits router."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--dataset", type=Path, help="Defaults to the active policy dataset.")
    parser.add_argument(
        "--regression-manifest",
        type=Path,
        default=Path("research/evals/router_regression_slices.json"),
    )
    parser.add_argument("--margins", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
    args = parser.parse_args()

    selection = select_threshold(
        args.registry,
        args.dataset,
        args.regression_manifest,
        args.margins,
        max_attempts=args.max_attempts,
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
    args.policy_output.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selection": selection, "policy": policy}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
