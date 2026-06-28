from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.latency_safe_probe import sweep_probe_policies
from mempool.routing_dataset import read_routing_records, validate_routing_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate cheap probe policies for detecting latency-safe routing rows."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-probe-count", type=int, default=2)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = read_routing_records(args.dataset)
    errors = validate_routing_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    policies = sweep_probe_policies(
        records,
        max_probe_count=args.max_probe_count,
        min_pass_rate=args.min_pass_rate,
    )
    payload = {
        "valid": True,
        "dataset": str(args.dataset),
        "max_probe_count": args.max_probe_count,
        "min_pass_rate": args.min_pass_rate,
        "policy_count": len(policies),
        "best_policies": policies[:20],
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "valid": True,
                "dataset": str(args.dataset),
                "policy_count": len(policies),
                "best_policy": {
                    key: policies[0][key]
                    for key in [
                        "probe_worker_ids",
                        "mode",
                        "accuracy",
                        "precision",
                        "recall",
                        "positive_count",
                        "predicted_positive_count",
                        "mean_probe_latency_ms",
                        "tp",
                        "fp",
                        "tn",
                        "fn",
                    ]
                }
                if policies
                else None,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
