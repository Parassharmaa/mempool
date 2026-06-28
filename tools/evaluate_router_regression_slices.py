from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.conditional_policy import evaluate_conditional_fallback, evaluate_gated_fallback
from mempool.logits_router import evaluate_logits_router
from mempool.routing_dataset import read_routing_records, validate_routing_records

try:
    from tools.evaluate_active_policy import load_active_router
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_slices(
    registry: Path,
    manifest: Path,
    policy: str = "logits",
    max_attempts: int = 2,
    max_first_second_margin: float = 0.1,
) -> dict[str, Any]:
    router, active = load_active_router(registry)
    payload = read_json(manifest)
    results = []
    for item in payload.get("slices", []):
        dataset = Path(item["dataset"])
        records = read_routing_records(dataset)
        errors = validate_routing_records(records)
        if errors:
            results.append(
                {
                    "id": item["id"],
                    "dataset": str(dataset),
                    "passed": False,
                    "errors": errors,
                }
            )
            continue
        if policy == "logits":
            evaluation = evaluate_logits_router(records, router)
        elif policy == "conditional":
            evaluation = evaluate_conditional_fallback(records, router, max_attempts=max_attempts)
        elif policy == "gated":
            evaluation = evaluate_gated_fallback(
                records,
                router,
                max_attempts=max_attempts,
                max_first_second_margin=max_first_second_margin,
            )
        else:
            raise ValueError(f"unknown policy: {policy}")
        min_solvable = float(item.get("minimum_solvable_pass_at_1", 0.0))
        expected_count = item.get("expected_solvable_task_count")
        count_ok = (
            expected_count is None
            or int(evaluation["solvable_task_count"]) == int(expected_count)
        )
        metric_ok = float(evaluation["solvable_pass_at_1"]) >= min_solvable
        results.append(
            {
                "id": item["id"],
                "dataset": str(dataset),
                "note": item.get("note"),
                "passed": count_ok and metric_ok,
                "minimum_solvable_pass_at_1": min_solvable,
                "expected_solvable_task_count": expected_count,
                "evaluation": evaluation,
            }
        )
    return {
        "registry": str(registry),
        "manifest": str(manifest),
        "policy": policy,
        "max_attempts": max_attempts if policy in {"conditional", "gated"} else None,
        "max_first_second_margin": max_first_second_margin if policy == "gated" else None,
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "passed": all(item["passed"] for item in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate active router against regression slices."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--manifest", type=Path, default=Path("research/evals/router_regression_slices.json"))
    parser.add_argument("--policy", choices=["logits", "conditional", "gated"], default="logits")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-first-second-margin", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate_slices(
        args.registry,
        args.manifest,
        policy=args.policy,
        max_attempts=args.max_attempts,
        max_first_second_margin=args.max_first_second_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
