from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.conditional_policy import evaluate_conditional_fallback, evaluate_gated_fallback
from mempool.routing_dataset import read_routing_records, validate_routing_records

try:
    from tools.evaluate_active_policy import load_active_router
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate conditional verifier/fallback behavior for the active router."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["always", "gated"], default="always")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-first-second-margin", type=float, default=0.1)
    args = parser.parse_args()

    router, active = load_active_router(args.registry)
    records = read_routing_records(args.dataset)
    errors = validate_routing_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    if args.policy == "always":
        evaluation = evaluate_conditional_fallback(records, router, max_attempts=args.max_attempts)
    else:
        evaluation = evaluate_gated_fallback(
            records,
            router,
            max_attempts=args.max_attempts,
            max_first_second_margin=args.max_first_second_margin,
        )
    payload = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "evaluated_dataset": str(args.dataset),
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
