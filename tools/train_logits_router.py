from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.logits_router import (
    evaluate_logits_router,
    leave_one_out_logits_evaluation,
    train_logits_router,
)
from mempool.routing_dataset import read_routing_records, validate_routing_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tiny logits-head router.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument(
        "--target-mode",
        choices=["distribution", "reward"],
        default="distribution",
        help="Train against stored target_distribution or a softmax over worker rewards.",
    )
    parser.add_argument(
        "--reward-temperature",
        type=float,
        default=0.25,
        help="Softmax temperature when --target-mode reward is selected.",
    )
    args = parser.parse_args()

    records = read_routing_records(args.dataset)
    errors = validate_routing_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    router, history = train_logits_router(
        records,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        target_mode=args.target_mode,
        reward_temperature=args.reward_temperature,
    )
    evaluation = evaluate_logits_router(records, router)
    leave_one_out = leave_one_out_logits_evaluation(
        records,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        target_mode=args.target_mode,
        reward_temperature=args.reward_temperature,
    )
    model_payload = {
        "model_type": "linear-softmax-logits-router",
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "target_mode": args.target_mode,
        "reward_temperature": args.reward_temperature,
        "router": router.to_dict(),
    }
    report_payload = {
        "dataset": str(args.dataset),
        "model_output": str(args.model_output),
        "target_mode": args.target_mode,
        "reward_temperature": args.reward_temperature,
        "history": history,
        "evaluation": evaluation,
        "leave_one_out": leave_one_out,
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(
        json.dumps(model_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
