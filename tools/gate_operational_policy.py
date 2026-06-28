from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.policy_refresh_gate import (
        EPSILON,
        policy_evaluation_metrics,
        read_json,
        strongest_worker_pass_at_1,
    )
except ModuleNotFoundError:
    from policy_refresh_gate import (
        EPSILON,
        policy_evaluation_metrics,
        read_json,
        strongest_worker_pass_at_1,
    )


def evaluation_metrics(report: dict[str, Any], policy_name: str | None = None) -> dict[str, Any]:
    if policy_name:
        metrics = policy_evaluation_metrics(report, policy_name)
        if metrics["available"]:
            return metrics
    evaluation = report.get("evaluation") or report.get("active_evaluation")
    if isinstance(evaluation, dict):
        return {
            "available": True,
            "policy": str(evaluation.get("policy", policy_name or "candidate")),
            "target_accuracy": float(evaluation.get("target_accuracy", 0.0)),
            "pass_at_1": float(evaluation.get("pass_at_1", 0.0)),
            "solvable_task_count": int(evaluation.get("solvable_task_count", 0)),
            "solvable_pass_at_1": float(evaluation.get("solvable_pass_at_1", 0.0)),
            "solvable_target_accuracy": float(
                evaluation.get("solvable_target_accuracy", 0.0)
            ),
            "mean_latency_regret_ms": float(
                evaluation.get("mean_latency_regret_ms", 0.0)
            ),
            "mean_latency_ms": float(evaluation.get("mean_latency_ms", 0.0)),
            "task_count": int(evaluation.get("task_count", 0)),
        }
    return {
        "available": False,
        "policy": policy_name or "candidate",
        "reason": "no named policy evaluation or top-level evaluation found",
    }


def gate_operational_policy(
    candidate_report: dict[str, Any],
    reference_report: dict[str, Any],
    *,
    candidate_policy: str | None = None,
    reference_policy: str,
    dataset: Path | None = None,
    min_pass_at_1_delta: float = 0.0,
    min_solvable_pass_at_1_delta: float = 0.0,
    min_target_accuracy_delta: float = 0.0,
    max_latency_regret_increase_ms: float = 0.0,
    min_pass_at_1_vs_strongest: float | None = None,
) -> dict[str, Any]:
    candidate = evaluation_metrics(candidate_report, candidate_policy)
    reference = policy_evaluation_metrics(reference_report, reference_policy)
    strongest_pass = strongest_worker_pass_at_1(dataset) if dataset else None
    reasons: list[str] = []
    warnings: list[str] = []

    if not candidate.get("available", False):
        reasons.append(f"candidate unavailable: {candidate.get('reason', 'unknown reason')}")
    if not reference.get("available", False):
        reasons.append(f"reference unavailable: {reference.get('reason', 'unknown reason')}")

    if candidate.get("available", False) and reference.get("available", False):
        pass_delta = float(candidate["pass_at_1"]) - float(reference["pass_at_1"])
        if pass_delta + EPSILON < min_pass_at_1_delta:
            reasons.append(
                f"candidate pass@1 delta {pass_delta:.3f} is below required "
                f"{min_pass_at_1_delta:.3f} against {reference_policy}"
            )

        solvable_delta = float(candidate["solvable_pass_at_1"]) - float(
            reference["solvable_pass_at_1"]
        )
        if solvable_delta + EPSILON < min_solvable_pass_at_1_delta:
            reasons.append(
                f"candidate solvable pass@1 delta {solvable_delta:.3f} is below "
                f"required {min_solvable_pass_at_1_delta:.3f} against {reference_policy}"
            )

        target_delta = float(candidate["target_accuracy"]) - float(
            reference["target_accuracy"]
        )
        if target_delta + EPSILON < min_target_accuracy_delta:
            reasons.append(
                f"candidate target accuracy delta {target_delta:.3f} is below "
                f"required {min_target_accuracy_delta:.3f} against {reference_policy}"
            )
        elif target_delta < 0:
            warnings.append(
                f"candidate target accuracy dropped by {-target_delta:.3f} "
                f"against {reference_policy}"
            )

        regret_increase = float(candidate["mean_latency_regret_ms"]) - float(
            reference["mean_latency_regret_ms"]
        )
        if regret_increase > max_latency_regret_increase_ms + EPSILON:
            reasons.append(
                f"candidate latency regret increase {regret_increase:.1f} ms "
                f"exceeds allowed {max_latency_regret_increase_ms:.1f} ms "
                f"against {reference_policy}"
            )
        elif regret_increase > 0:
            warnings.append(
                f"candidate latency regret increased by {regret_increase:.1f} ms "
                f"against {reference_policy}"
            )

    if (
        candidate.get("available", False)
        and strongest_pass is not None
        and min_pass_at_1_vs_strongest is not None
        and float(candidate["pass_at_1"])
        < strongest_pass + min_pass_at_1_vs_strongest - EPSILON
    ):
        reasons.append(
            f"candidate pass@1 {float(candidate['pass_at_1']):.3f} is below "
            f"strongest-worker pass@1 {strongest_pass:.3f} plus required "
            f"delta {min_pass_at_1_vs_strongest:.3f}"
        )

    return {
        "decision": "promote" if not reasons else "quarantine",
        "timestamp": datetime.now(UTC).isoformat(),
        "candidate": candidate,
        "reference": reference,
        "dataset": str(dataset) if dataset else None,
        "strongest_worker": {"pass_at_1": strongest_pass},
        "thresholds": {
            "min_pass_at_1_delta": min_pass_at_1_delta,
            "min_solvable_pass_at_1_delta": min_solvable_pass_at_1_delta,
            "min_target_accuracy_delta": min_target_accuracy_delta,
            "max_latency_regret_increase_ms": max_latency_regret_increase_ms,
            "min_pass_at_1_vs_strongest": min_pass_at_1_vs_strongest,
        },
        "reasons": reasons,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate an evaluated operational policy against a reference policy."
    )
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-policy")
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-policy", required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pass-at-1-delta", type=float, default=0.0)
    parser.add_argument("--min-solvable-pass-at-1-delta", type=float, default=0.0)
    parser.add_argument("--min-target-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--max-latency-regret-increase-ms", type=float, default=0.0)
    parser.add_argument("--min-pass-at-1-vs-strongest", type=float)
    args = parser.parse_args()

    result = gate_operational_policy(
        read_json(args.candidate_report),
        read_json(args.reference_report),
        candidate_policy=args.candidate_policy,
        reference_policy=args.reference_policy,
        dataset=args.dataset,
        min_pass_at_1_delta=args.min_pass_at_1_delta,
        min_solvable_pass_at_1_delta=args.min_solvable_pass_at_1_delta,
        min_target_accuracy_delta=args.min_target_accuracy_delta,
        max_latency_regret_increase_ms=args.max_latency_regret_increase_ms,
        min_pass_at_1_vs_strongest=args.min_pass_at_1_vs_strongest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "promote" else 2


if __name__ == "__main__":
    raise SystemExit(main())
