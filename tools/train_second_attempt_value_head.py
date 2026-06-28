from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.multi_head_orchestrator import read_substrate, validate_substrate_records
from mempool.second_attempt_value import (
    SecondAttemptValueHead,
    evaluate_learned_value_head,
    leave_one_out_value_head_evaluation,
    rank_value_head_evaluation,
    train_second_attempt_value_head,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a learned second-attempt value head from multi-head LOO predictions."
    )
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--latency-cost-per-second", type=float, default=0.01)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.0001)
    args = parser.parse_args()

    records = read_substrate(args.substrate)
    errors = validate_substrate_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    source_report = read_json(args.report)
    predictions = (source_report.get("leave_one_out") or {}).get("predictions") or []
    if not predictions:
        print(json.dumps({"valid": False, "errors": ["missing leave-one-out predictions"]}, indent=2))
        return 1

    base_head, history = train_second_attempt_value_head(
        records,
        predictions,
        latency_cost_per_second=args.latency_cost_per_second,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    candidates = []
    for threshold in args.thresholds:
        head = SecondAttemptValueHead(
            feature_names=base_head.feature_names,
            weights=base_head.weights,
            threshold=threshold,
        )
        candidates.append(
            {
                "threshold": threshold,
                "evaluation": evaluate_learned_value_head(
                    records,
                    predictions,
                    head,
                    latency_cost_per_second=args.latency_cost_per_second,
                ),
            }
        )
    selected = max(
        candidates,
        key=lambda candidate: rank_value_head_evaluation(candidate["evaluation"]),
    )
    selected_head = SecondAttemptValueHead(
        feature_names=base_head.feature_names,
        weights=base_head.weights,
        threshold=selected["threshold"],
    )
    model_payload = {
        "model_type": "second-attempt-value-head",
        "substrate": str(args.substrate),
        "source_report": str(args.report),
        "latency_cost_per_second": args.latency_cost_per_second,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "head": selected_head.to_dict(),
    }
    report_payload = {
        "model_output": str(args.model_output),
        "substrate": str(args.substrate),
        "source_report": str(args.report),
        "latency_cost_per_second": args.latency_cost_per_second,
        "history": history,
        "selected": selected,
        "evaluation": selected["evaluation"],
        "in_sample_leave_one_out_predictions": {"available": True, **selected["evaluation"]},
        "leave_one_out": leave_one_out_value_head_evaluation(
            records,
            predictions,
            latency_cost_per_second=args.latency_cost_per_second,
            thresholds=args.thresholds,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        ),
        "candidates": candidates,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_output.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
