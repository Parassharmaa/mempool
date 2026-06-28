from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.logits_router import (
    LogitsRouter,
    evaluate_logits_router,
    leave_one_out_logits_evaluation,
    train_logits_router,
)
from mempool.routing_dataset import read_routing_records, validate_routing_records

try:
    from tools.policy_refresh_gate import (
        evaluate_refresh,
        policy_evaluation_metrics,
        read_json,
        thresholds_for_profile,
    )
except ModuleNotFoundError:
    from policy_refresh_gate import (
        evaluate_refresh,
        policy_evaluation_metrics,
        read_json,
        thresholds_for_profile,
    )


def temperature_slug(temperature: float) -> str:
    text = f"{temperature:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    loo = candidate["refresh"]["candidate"]["loo"]
    return (
        float(loo["target_accuracy"]),
        float(loo.get("solvable_pass_at_1", 0.0)),
        float(loo["pass_at_1"]),
        -float(loo["mean_latency_regret_ms"]),
        -float(loo["mean_kl"]),
    )


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    promotable = [
        candidate
        for candidate in candidates
        if candidate["refresh"]["decision"] == "promote"
    ]
    if not promotable:
        return None
    return max(promotable, key=candidate_rank)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_temperature_candidate(
    records: list[dict[str, Any]],
    dataset: Path,
    model_output: Path,
    report_output: Path,
    temperature: float,
    epochs: int,
    learning_rate: float,
    l2: float,
    initial_router: LogitsRouter | None = None,
    initial_model: Path | None = None,
) -> dict[str, Any]:
    router, history = train_logits_router(
        records,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        target_mode="reward",
        reward_temperature=temperature,
        initial_router=initial_router,
    )
    evaluation = evaluate_logits_router(records, router)
    leave_one_out = leave_one_out_logits_evaluation(
        records,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        target_mode="reward",
        reward_temperature=temperature,
        initial_router=initial_router,
    )
    model_payload = {
        "model_type": "linear-softmax-logits-router",
        "dataset": str(dataset),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "target_mode": "reward",
        "reward_temperature": temperature,
        "initial_model": str(initial_model) if initial_model else None,
        "router": router.to_dict(),
    }
    report_payload = {
        "dataset": str(dataset),
        "model_output": str(model_output),
        "target_mode": "reward",
        "reward_temperature": temperature,
        "initial_model": str(initial_model) if initial_model else None,
        "history": history,
        "evaluation": evaluation,
        "leave_one_out": leave_one_out,
    }
    write_json(model_output, model_payload)
    write_json(report_output, report_payload)
    return report_payload


def run_selection(
    records: list[dict[str, Any]],
    dataset: Path,
    temperatures: list[float],
    model_dir: Path,
    report_dir: Path,
    prefix: str,
    epochs: int,
    learning_rate: float,
    l2: float,
    baseline_report: dict[str, Any] | None,
    baseline_dataset: Path | None,
    operational_reference: dict[str, Any] | None,
    min_loo_accuracy: float,
    max_loo_accuracy_drop: float,
    min_loo_solvable_pass_at_1: float | None,
    max_loo_latency_regret_ms: float | None,
    max_loo_latency_regret_increase_ms: float | None,
    promotion_profile: str,
    initial_router: LogitsRouter | None = None,
    initial_model: Path | None = None,
    min_loo_pass_at_1_vs_strongest: float | None = None,
) -> dict[str, Any]:
    candidates = []
    thresholds = thresholds_for_profile(
        profile=promotion_profile,
        baseline_report=baseline_report,
        operational_reference=operational_reference,
        min_loo_accuracy=min_loo_accuracy,
        max_loo_accuracy_drop=max_loo_accuracy_drop,
        min_loo_solvable_pass_at_1=min_loo_solvable_pass_at_1,
        max_loo_latency_regret_ms=max_loo_latency_regret_ms,
        max_loo_latency_regret_increase_ms=max_loo_latency_regret_increase_ms,
        min_loo_pass_at_1_vs_strongest=min_loo_pass_at_1_vs_strongest,
    )
    for temperature in temperatures:
        slug = temperature_slug(temperature)
        model_output = model_dir / f"{prefix}-reward-t{slug}-logits-router.json"
        report_output = report_dir / f"{prefix}-reward-t{slug}-logits-router-report.json"
        report = train_temperature_candidate(
            records=records,
            dataset=dataset,
            model_output=model_output,
            report_output=report_output,
            temperature=temperature,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            initial_router=initial_router,
            initial_model=initial_model,
        )
        refresh = evaluate_refresh(
            candidate_report=report,
            candidate_dataset=dataset,
            baseline_report=baseline_report,
            baseline_dataset=baseline_dataset,
            operational_reference=operational_reference,
            min_loo_accuracy=thresholds["min_loo_accuracy"],
            max_loo_accuracy_drop=thresholds["max_loo_accuracy_drop"],
            min_loo_solvable_pass_at_1=thresholds["min_loo_solvable_pass_at_1"],
            max_loo_latency_regret_ms=thresholds["max_loo_latency_regret_ms"],
            max_loo_latency_regret_increase_ms=thresholds["max_loo_latency_regret_increase_ms"],
            min_loo_pass_at_1_vs_strongest=thresholds["min_loo_pass_at_1_vs_strongest"],
        )
        refresh["promotion_profile"] = promotion_profile
        candidates.append(
            {
                "temperature": temperature,
                "model_output": str(model_output),
                "report_output": str(report_output),
                "refresh": refresh,
            }
        )

    selected = select_best_candidate(candidates)
    return {
        "dataset": str(dataset),
        "target_mode": "reward",
        "promotion_profile": promotion_profile,
        "initial_model": str(initial_model) if initial_model else None,
        "temperatures": temperatures,
        "candidates": candidates,
        "selected": selected,
        "decision": "promote" if selected else "quarantine",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep reward temperatures and select a gated logits-router candidate."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("research/models"))
    parser.add_argument("--report-dir", type=Path, default=Path("research/datasets"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--refresh-output", type=Path)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.2, 0.5])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--baseline-dataset", type=Path)
    parser.add_argument("--operational-baseline-report", type=Path)
    parser.add_argument("--operational-baseline-policy")
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--min-loo-accuracy", type=float, default=0.7)
    parser.add_argument("--max-loo-accuracy-drop", type=float, default=0.1)
    parser.add_argument("--min-loo-solvable-pass-at-1", type=float)
    parser.add_argument("--max-loo-latency-regret-ms", type=float)
    parser.add_argument("--max-loo-latency-regret-increase-ms", type=float)
    parser.add_argument("--min-loo-pass-at-1-vs-strongest", type=float)
    parser.add_argument(
        "--promotion-profile",
        choices=["tolerant", "preserve_accuracy"],
        default="tolerant",
    )
    args = parser.parse_args()

    records = read_routing_records(args.dataset)
    errors = validate_routing_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    baseline_report = read_json(args.baseline_report) if args.baseline_report else None
    operational_reference = None
    if args.operational_baseline_report and args.operational_baseline_policy:
        operational_reference = policy_evaluation_metrics(
            read_json(args.operational_baseline_report),
            args.operational_baseline_policy,
        )
    initial_router = None
    if args.initial_model:
        initial_model_payload = read_json(args.initial_model)
        initial_router = LogitsRouter.from_dict(initial_model_payload["router"])
    selection = run_selection(
        records=records,
        dataset=args.dataset,
        temperatures=args.temperatures,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        prefix=args.prefix,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        baseline_report=baseline_report,
        baseline_dataset=args.baseline_dataset,
        operational_reference=operational_reference,
        min_loo_accuracy=args.min_loo_accuracy,
        max_loo_accuracy_drop=args.max_loo_accuracy_drop,
        min_loo_solvable_pass_at_1=args.min_loo_solvable_pass_at_1,
        max_loo_latency_regret_ms=args.max_loo_latency_regret_ms,
        max_loo_latency_regret_increase_ms=args.max_loo_latency_regret_increase_ms,
        min_loo_pass_at_1_vs_strongest=args.min_loo_pass_at_1_vs_strongest,
        promotion_profile=args.promotion_profile,
        initial_router=initial_router,
        initial_model=args.initial_model,
    )
    write_json(args.selection_output, selection)
    if args.refresh_output and selection["selected"]:
        write_json(args.refresh_output, selection["selected"]["refresh"])
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0 if selection["selected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
