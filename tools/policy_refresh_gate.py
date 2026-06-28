from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EPSILON = 1e-9
PROMOTION_PROFILES = ("tolerant", "preserve_accuracy")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def loo_metrics(report: dict[str, Any]) -> dict[str, Any]:
    loo = report.get("leave_one_out") or {}
    if loo.get("available") is False:
        return {"available": False}
    return {
        "available": bool(loo),
        "target_accuracy": float(loo.get("target_accuracy", 0.0)),
        "pass_at_1": float(loo.get("pass_at_1", 0.0)),
        "solvable_task_count": int(loo.get("solvable_task_count", 0)),
        "solvable_pass_at_1": float(loo.get("solvable_pass_at_1", 0.0)),
        "solvable_target_accuracy": float(loo.get("solvable_target_accuracy", 0.0)),
        "mean_kl": float(loo.get("mean_kl", 0.0)),
        "mean_latency_regret_ms": float(loo.get("mean_latency_regret_ms", 0.0)),
        "task_count": int(loo.get("task_count", 0)),
    }


def policy_evaluation_metrics(report: dict[str, Any], policy_name: str) -> dict[str, Any]:
    for evaluation in report.get("evaluations", []):
        if evaluation.get("policy") == policy_name:
            return {
                "available": True,
                "policy": policy_name,
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
                "task_count": int(evaluation.get("task_count", 0)),
            }
    return {
        "available": False,
        "policy": policy_name,
        "reason": f"policy evaluation not found: {policy_name}",
    }


def target_mix(dataset_path: Path) -> dict[str, Any]:
    records = read_jsonl(dataset_path)
    counts = Counter(record["target_worker_id"] for record in records)
    return {
        "task_count": len(records),
        "target_worker_count": len(counts),
        "target_counts": dict(sorted(counts.items())),
    }


def strongest_worker_pass_at_1(dataset_path: Path) -> float:
    records = read_jsonl(dataset_path)
    if not records:
        return 0.0
    worker_ids = sorted(
        {
            str(worker["worker_id"])
            for record in records
            for worker in record.get("workers", [])
        }
    )
    if not worker_ids:
        return 0.0
    scores = {}
    for worker_id in worker_ids:
        solved = 0
        available = 0
        for record in records:
            worker = next(
                (
                    item
                    for item in record.get("workers", [])
                    if str(item["worker_id"]) == worker_id
                ),
                None,
            )
            if worker is None:
                continue
            available += 1
            solved += int(bool(worker.get("passed")))
        scores[worker_id] = solved / available if available else 0.0
    return max(scores.values(), default=0.0)


def evaluate_refresh(
    candidate_report: dict[str, Any],
    candidate_dataset: Path,
    baseline_report: dict[str, Any] | None = None,
    baseline_dataset: Path | None = None,
    operational_reference: dict[str, Any] | None = None,
    min_loo_accuracy: float = 0.7,
    max_loo_accuracy_drop: float = 0.1,
    min_loo_solvable_pass_at_1: float | None = None,
    max_loo_latency_regret_ms: float | None = None,
    max_loo_latency_regret_increase_ms: float | None = None,
    min_loo_pass_at_1_vs_strongest: float | None = None,
) -> dict[str, Any]:
    candidate_loo = loo_metrics(candidate_report)
    candidate_mix = target_mix(candidate_dataset)
    baseline_loo = loo_metrics(baseline_report) if baseline_report else None
    baseline_mix = target_mix(baseline_dataset) if baseline_dataset else None
    operational_reference = operational_reference or None
    strongest_pass_at_1 = strongest_worker_pass_at_1(candidate_dataset)

    reasons = []
    warnings = []
    if not candidate_loo["available"]:
        reasons.append("candidate leave-one-out evaluation is unavailable")
    elif candidate_loo["target_accuracy"] < min_loo_accuracy:
        reasons.append(
            f"candidate LOO target accuracy {candidate_loo['target_accuracy']:.3f} "
            f"is below minimum {min_loo_accuracy:.3f}"
        )
    elif (
        min_loo_solvable_pass_at_1 is not None
        and candidate_loo["solvable_task_count"] > 0
        and candidate_loo["solvable_pass_at_1"] < min_loo_solvable_pass_at_1
    ):
        reasons.append(
            f"candidate LOO solvable pass@1 {candidate_loo['solvable_pass_at_1']:.3f} "
            f"is below minimum {min_loo_solvable_pass_at_1:.3f}"
        )

    if baseline_loo and baseline_loo["available"]:
        drop = baseline_loo["target_accuracy"] - candidate_loo["target_accuracy"]
        if drop > max_loo_accuracy_drop + EPSILON:
            reasons.append(
                f"candidate LOO target accuracy drop {drop:.3f} exceeds "
                f"allowed {max_loo_accuracy_drop:.3f}"
            )
        elif drop > 0:
            warnings.append(f"candidate LOO target accuracy dropped by {drop:.3f}")

        if max_loo_latency_regret_increase_ms is not None:
            regret_increase = (
                candidate_loo["mean_latency_regret_ms"]
                - baseline_loo["mean_latency_regret_ms"]
            )
            if regret_increase > max_loo_latency_regret_increase_ms + EPSILON:
                reasons.append(
                    f"candidate LOO mean latency regret increase "
                    f"{regret_increase:.1f} ms exceeds allowed "
                    f"{max_loo_latency_regret_increase_ms:.1f} ms"
                )
            elif regret_increase > 0:
                warnings.append(
                    f"candidate LOO mean latency regret increased by "
                    f"{regret_increase:.1f} ms"
                )

    if operational_reference:
        if not operational_reference.get("available", False):
            reasons.append(
                f"operational reference unavailable: "
                f"{operational_reference.get('reason', 'unknown reason')}"
            )
        elif candidate_loo["available"]:
            operational_drop = (
                float(operational_reference["target_accuracy"])
                - candidate_loo["target_accuracy"]
            )
            if operational_drop > max_loo_accuracy_drop + EPSILON:
                reasons.append(
                    f"candidate LOO target accuracy drop {operational_drop:.3f} "
                    f"against operational reference "
                    f"{operational_reference.get('policy', 'unknown')} exceeds "
                    f"allowed {max_loo_accuracy_drop:.3f}"
                )
            elif operational_drop > 0:
                warnings.append(
                    f"candidate LOO target accuracy dropped by "
                    f"{operational_drop:.3f} against operational reference "
                    f"{operational_reference.get('policy', 'unknown')}"
                )

            if max_loo_latency_regret_increase_ms is not None:
                operational_regret_increase = (
                    candidate_loo["mean_latency_regret_ms"]
                    - float(operational_reference["mean_latency_regret_ms"])
                )
                if (
                    operational_regret_increase
                    > max_loo_latency_regret_increase_ms + EPSILON
                ):
                    reasons.append(
                        f"candidate LOO mean latency regret increase "
                        f"{operational_regret_increase:.1f} ms against operational "
                        f"reference {operational_reference.get('policy', 'unknown')} "
                        f"exceeds allowed "
                        f"{max_loo_latency_regret_increase_ms:.1f} ms"
                    )
                elif operational_regret_increase > 0:
                    warnings.append(
                        f"candidate LOO mean latency regret increased by "
                        f"{operational_regret_increase:.1f} ms against operational "
                        f"reference {operational_reference.get('policy', 'unknown')}"
                    )

    if (
        candidate_loo["available"]
        and max_loo_latency_regret_ms is not None
        and candidate_loo["mean_latency_regret_ms"] > max_loo_latency_regret_ms + EPSILON
    ):
        reasons.append(
            f"candidate LOO mean latency regret "
            f"{candidate_loo['mean_latency_regret_ms']:.1f} ms exceeds maximum "
            f"{max_loo_latency_regret_ms:.1f} ms"
        )

    if baseline_mix:
        if candidate_mix["task_count"] < baseline_mix["task_count"]:
            reasons.append("candidate dataset has fewer tasks than baseline")
        if candidate_mix["target_worker_count"] < baseline_mix["target_worker_count"]:
            reasons.append("candidate target-worker diversity regressed")

    if (
        candidate_loo["available"]
        and min_loo_pass_at_1_vs_strongest is not None
        and candidate_loo["pass_at_1"]
        < strongest_pass_at_1 + min_loo_pass_at_1_vs_strongest - EPSILON
    ):
        reasons.append(
            f"candidate LOO pass@1 {candidate_loo['pass_at_1']:.3f} is below "
            f"strongest-worker pass@1 {strongest_pass_at_1:.3f} plus required "
            f"delta {min_loo_pass_at_1_vs_strongest:.3f}"
        )

    decision = "promote" if not reasons else "quarantine"
    return {
        "decision": decision,
        "timestamp": datetime.now(UTC).isoformat(),
        "candidate": {
            "report": candidate_report.get("model_output"),
            "dataset": str(candidate_dataset),
            "loo": candidate_loo,
            "target_mix": candidate_mix,
        },
        "baseline": {
            "available": baseline_report is not None,
            "report": baseline_report.get("model_output") if baseline_report else None,
            "dataset": str(baseline_dataset) if baseline_dataset else None,
            "loo": baseline_loo,
            "target_mix": baseline_mix,
        },
        "operational_reference": operational_reference,
        "thresholds": {
            "min_loo_accuracy": min_loo_accuracy,
            "max_loo_accuracy_drop": max_loo_accuracy_drop,
            "min_loo_solvable_pass_at_1": min_loo_solvable_pass_at_1,
            "max_loo_latency_regret_ms": max_loo_latency_regret_ms,
            "max_loo_latency_regret_increase_ms": max_loo_latency_regret_increase_ms,
            "min_loo_pass_at_1_vs_strongest": min_loo_pass_at_1_vs_strongest,
        },
        "strongest_worker": {
            "pass_at_1": strongest_pass_at_1,
        },
        "reasons": reasons,
        "warnings": warnings,
    }


def thresholds_for_profile(
    *,
    profile: str,
    baseline_report: dict[str, Any] | None,
    min_loo_accuracy: float,
    max_loo_accuracy_drop: float,
    min_loo_solvable_pass_at_1: float | None,
    max_loo_latency_regret_ms: float | None,
    max_loo_latency_regret_increase_ms: float | None,
    min_loo_pass_at_1_vs_strongest: float | None = None,
    operational_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if profile not in PROMOTION_PROFILES:
        raise ValueError(f"unknown promotion profile: {profile}")
    thresholds = {
        "min_loo_accuracy": min_loo_accuracy,
        "max_loo_accuracy_drop": max_loo_accuracy_drop,
        "min_loo_solvable_pass_at_1": min_loo_solvable_pass_at_1,
        "max_loo_latency_regret_ms": max_loo_latency_regret_ms,
        "max_loo_latency_regret_increase_ms": max_loo_latency_regret_increase_ms,
        "min_loo_pass_at_1_vs_strongest": min_loo_pass_at_1_vs_strongest,
    }
    if profile == "preserve_accuracy" and baseline_report:
        baseline_loo = loo_metrics(baseline_report)
        if baseline_loo.get("available"):
            thresholds["min_loo_accuracy"] = max(
                thresholds["min_loo_accuracy"],
                float(baseline_loo["target_accuracy"]),
            )
            thresholds["max_loo_accuracy_drop"] = 0.0
            if baseline_loo["solvable_task_count"] > 0:
                thresholds["min_loo_solvable_pass_at_1"] = max(
                    thresholds["min_loo_solvable_pass_at_1"] or 0.0,
                    float(baseline_loo["solvable_pass_at_1"]),
                )
            thresholds["max_loo_latency_regret_ms"] = min(
                value
                for value in (
                    thresholds["max_loo_latency_regret_ms"],
                    float(baseline_loo["mean_latency_regret_ms"]),
                )
                if value is not None
            )
    if (
        profile == "preserve_accuracy"
        and operational_reference
        and operational_reference.get("available", False)
    ):
        thresholds["min_loo_accuracy"] = max(
            thresholds["min_loo_accuracy"],
            float(operational_reference["target_accuracy"]),
        )
        thresholds["max_loo_accuracy_drop"] = 0.0
        if int(operational_reference.get("solvable_task_count", 0)) > 0:
            thresholds["min_loo_solvable_pass_at_1"] = max(
                thresholds["min_loo_solvable_pass_at_1"] or 0.0,
                float(operational_reference["solvable_pass_at_1"]),
            )
        thresholds["max_loo_latency_regret_ms"] = min(
            value
            for value in (
                thresholds["max_loo_latency_regret_ms"],
                float(operational_reference["mean_latency_regret_ms"]),
            )
            if value is not None
        )
    thresholds["promotion_profile"] = profile
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a logits-router policy refresh.")
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--baseline-dataset", type=Path)
    parser.add_argument("--operational-baseline-report", type=Path)
    parser.add_argument("--operational-baseline-policy")
    parser.add_argument("--min-loo-accuracy", type=float, default=0.7)
    parser.add_argument("--max-loo-accuracy-drop", type=float, default=0.1)
    parser.add_argument("--min-loo-solvable-pass-at-1", type=float)
    parser.add_argument("--max-loo-latency-regret-ms", type=float)
    parser.add_argument("--max-loo-latency-regret-increase-ms", type=float)
    parser.add_argument("--min-loo-pass-at-1-vs-strongest", type=float)
    parser.add_argument("--promotion-profile", choices=PROMOTION_PROFILES, default="tolerant")
    args = parser.parse_args()

    baseline_report = read_json(args.baseline_report) if args.baseline_report else None
    operational_reference = None
    if args.operational_baseline_report and args.operational_baseline_policy:
        operational_reference = policy_evaluation_metrics(
            read_json(args.operational_baseline_report),
            args.operational_baseline_policy,
        )
    thresholds = thresholds_for_profile(
        profile=args.promotion_profile,
        baseline_report=baseline_report,
        operational_reference=operational_reference,
        min_loo_accuracy=args.min_loo_accuracy,
        max_loo_accuracy_drop=args.max_loo_accuracy_drop,
        min_loo_solvable_pass_at_1=args.min_loo_solvable_pass_at_1,
        max_loo_latency_regret_ms=args.max_loo_latency_regret_ms,
        max_loo_latency_regret_increase_ms=args.max_loo_latency_regret_increase_ms,
        min_loo_pass_at_1_vs_strongest=args.min_loo_pass_at_1_vs_strongest,
    )
    result = evaluate_refresh(
        candidate_report=read_json(args.candidate_report),
        candidate_dataset=args.candidate_dataset,
        baseline_report=baseline_report,
        baseline_dataset=args.baseline_dataset,
        operational_reference=operational_reference,
        min_loo_accuracy=thresholds["min_loo_accuracy"],
        max_loo_accuracy_drop=thresholds["max_loo_accuracy_drop"],
        min_loo_solvable_pass_at_1=thresholds["min_loo_solvable_pass_at_1"],
        max_loo_latency_regret_ms=thresholds["max_loo_latency_regret_ms"],
        max_loo_latency_regret_increase_ms=thresholds["max_loo_latency_regret_increase_ms"],
        min_loo_pass_at_1_vs_strongest=thresholds["min_loo_pass_at_1_vs_strongest"],
    )
    result["promotion_profile"] = args.promotion_profile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "promote" else 2


if __name__ == "__main__":
    raise SystemExit(main())
