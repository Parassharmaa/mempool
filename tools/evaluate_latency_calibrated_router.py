from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.latency_calibrated_router import (
    evaluate_latency_calibrated_predictions,
    rank_latency_calibrated_evaluation,
)
from mempool.multi_head_orchestrator import read_substrate, validate_substrate_records


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def concise_evaluation(evaluation: dict) -> dict:
    return {
        key: evaluation.get(key)
        for key in [
            "policy",
            "latency_cost_per_second",
            "min_probability_ratio",
            "min_probability",
            "task_count",
            "target_accuracy",
            "pass_at_1",
            "solvable_pass_at_1",
            "mean_latency_ms",
            "mean_target_latency_ms",
            "mean_latency_regret_ms",
            "changed_from_top",
            "change_rate",
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a transparent latency-calibrated worker choice layer."
    )
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--latency-costs", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02])
    parser.add_argument("--min-probability-ratios", type=float, nargs="+", default=[0.0, 0.5, 0.75, 0.9])
    parser.add_argument("--min-probabilities", type=float, nargs="+", default=[0.0])
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = read_substrate(args.substrate)
    errors = validate_substrate_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    source_report = read_json(args.report)
    loo = source_report.get("leave_one_out") or {}
    predictions = loo.get("predictions") or []
    if not predictions:
        print(json.dumps({"valid": False, "errors": ["missing leave-one-out predictions"]}, indent=2))
        return 1

    candidates = []
    for latency_cost in args.latency_costs:
        for min_probability_ratio in args.min_probability_ratios:
            for min_probability in args.min_probabilities:
                candidates.append(
                    evaluate_latency_calibrated_predictions(
                        records,
                        predictions,
                        latency_cost_per_second=latency_cost,
                        min_probability_ratio=min_probability_ratio,
                        min_probability=min_probability,
                    )
                )

    selected = max(candidates, key=rank_latency_calibrated_evaluation)
    model_output = str(args.model_output) if args.model_output else source_report.get("model_output")
    payload = {
        "valid": True,
        "model_output": model_output,
        "substrate": str(args.substrate),
        "report": str(args.report),
        "base_model": source_report.get("model_output"),
        "base_leave_one_out": {
            key: loo.get(key)
            for key in [
                "target_accuracy",
                "pass_at_1",
                "solvable_pass_at_1",
                "mean_latency_regret_ms",
                "mean_latency_ms",
                "mean_target_latency_ms",
            ]
        },
        "selected": selected,
        "leave_one_out": {"available": True, **selected},
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.model_output:
        model_payload = {
            "model_type": "latency-calibrated-multi-head-router",
            "base_model": source_report.get("model_output"),
            "source_report": str(args.report),
            "substrate": str(args.substrate),
            "calibration": {
                "policy": selected["policy"],
                "latency_cost_per_second": selected["latency_cost_per_second"],
                "min_probability_ratio": selected["min_probability_ratio"],
                "min_probability": selected["min_probability"],
            },
        }
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        args.model_output.write_text(
            json.dumps(model_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "valid": True,
                "substrate": payload["substrate"],
                "report": payload["report"],
                "base_leave_one_out": payload["base_leave_one_out"],
                "selected": concise_evaluation(selected),
                "candidate_count": len(candidates),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
