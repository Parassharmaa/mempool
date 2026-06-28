from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.multi_head_orchestrator import (
    evaluate_multi_head_orchestrator,
    leave_one_out_multi_head_evaluation,
    read_substrate,
    train_multi_head_orchestrator,
    validate_substrate_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a tiny local multi-head orchestrator on substrate JSONL."
    )
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument(
        "--latency-regret-weight",
        type=float,
        default=0.0,
        help="Optional expected latency-regret penalty for the worker head.",
    )
    args = parser.parse_args()

    records = read_substrate(args.substrate)
    errors = validate_substrate_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    model, history = train_multi_head_orchestrator(
        records,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        latency_regret_weight=args.latency_regret_weight,
    )
    evaluation = evaluate_multi_head_orchestrator(records, model)
    leave_one_out = leave_one_out_multi_head_evaluation(
        records,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    model_payload = {
        "model_type": "linear-multi-head-orchestrator",
        "substrate": str(args.substrate),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "latency_regret_weight": args.latency_regret_weight,
        "orchestrator": model.to_dict(),
    }
    report_payload = {
        "substrate": str(args.substrate),
        "model_output": str(args.model_output),
        "history": history,
        "latency_regret_weight": args.latency_regret_weight,
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
