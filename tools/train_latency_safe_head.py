from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.latency_safe_head import (
    evaluate_latency_safe_head,
    leave_one_out_latency_safe_evaluation,
    train_latency_safe_head,
)
from mempool.logits_router import LogitsRouter
from mempool.routing_dataset import read_routing_records, validate_routing_records


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a small classifier for rows that are safe for latency-only optimization."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--router-model", type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--use-reliability-features", action="store_true")
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    records = read_routing_records(args.dataset)
    errors = validate_routing_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    router = None
    if args.router_model:
        router = LogitsRouter.from_dict(read_json(args.router_model)["router"])
    head, history = train_latency_safe_head(
        records,
        router=router,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        threshold=args.threshold,
        positive_weight=args.positive_weight,
        use_reliability_features=args.use_reliability_features,
    )
    reliability_context = None
    if args.use_reliability_features:
        from mempool.latency_safe_head import worker_reliability_context

        reliability_context = worker_reliability_context(records)
    evaluation = evaluate_latency_safe_head(
        records,
        head,
        router=router,
        reliability_context=reliability_context,
    )
    leave_one_out = leave_one_out_latency_safe_evaluation(
        records,
        router=router,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        threshold=args.threshold,
        positive_weight=args.positive_weight,
        use_reliability_features=args.use_reliability_features,
    )
    model_payload = {
        "model_type": "latency-safe-logit-head",
        "dataset": str(args.dataset),
        "router_model": str(args.router_model) if args.router_model else None,
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "threshold": args.threshold,
            "positive_weight": args.positive_weight,
            "use_reliability_features": args.use_reliability_features,
        },
        "head": head.to_dict(),
    }
    report_payload = {
        "valid": True,
        "dataset": str(args.dataset),
        "router_model": str(args.router_model) if args.router_model else None,
        "model_output": str(args.model_output),
        "training": model_payload["training"],
        "feature_count": len(head.feature_names),
        "history": history,
        "evaluation": evaluation,
        "leave_one_out": leave_one_out,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_output.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "valid": True,
                "dataset": str(args.dataset),
                "router_model": str(args.router_model) if args.router_model else None,
                "feature_count": len(head.feature_names),
                "evaluation": {
                    key: evaluation[key]
                    for key in [
                        "task_count",
                        "accuracy",
                        "precision",
                        "recall",
                        "positive_count",
                        "predicted_positive_count",
                    ]
                },
                "leave_one_out": {
                    key: leave_one_out[key]
                    for key in [
                        "available",
                        "task_count",
                        "accuracy",
                        "precision",
                        "recall",
                        "positive_count",
                        "predicted_positive_count",
                    ]
                    if key in leave_one_out
                },
                "report_output": str(args.report_output),
                "model_output": str(args.model_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
