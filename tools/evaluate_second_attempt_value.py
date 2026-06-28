from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.multi_head_orchestrator import read_substrate, validate_substrate_records
from mempool.second_attempt_value import evaluate_value_gated_fallback


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate oracle second-attempt value gating for a multi-head report."
    )
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--latency-costs", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--min-values", type=float, nargs="+", default=[0.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = read_substrate(args.substrate)
    errors = validate_substrate_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    report = read_json(args.report)
    loo = report.get("leave_one_out") or {}
    predictions = loo.get("predictions") or []
    if not predictions:
        print(json.dumps({"valid": False, "errors": ["missing leave-one-out predictions"]}, indent=2))
        return 1

    candidates = []
    for cost in args.latency_costs:
        for min_value in args.min_values:
            candidates.append(
                evaluate_value_gated_fallback(
                    records,
                    predictions,
                    latency_cost_per_second=cost,
                    min_value=min_value,
                )
            )

    selected = max(
        candidates,
        key=lambda item: (
            item["solvable_pass_at_1"],
            item["pass_at_1"],
            item["target_accuracy"],
            -item["mean_latency_regret_ms"],
            -item["fallback_rate"],
        ),
    )
    payload = {
        "valid": True,
        "substrate": str(args.substrate),
        "report": str(args.report),
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
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
