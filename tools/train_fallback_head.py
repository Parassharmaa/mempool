from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.fallback_head import FallbackHead, evaluate_fallback_head, train_fallback_head
from mempool.routing_dataset import read_routing_records, validate_routing_records

try:
    from tools.evaluate_active_policy import load_active_router
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_regression_slices(
    registry: Path,
    manifest: Path,
    head: FallbackHead,
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
        evaluation = evaluate_fallback_head(records, router, head)
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
        "policy": "fallback-logit-head",
        "threshold": head.threshold,
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "passed": all(item["passed"] for item in results),
        "results": results,
    }


def rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    evaluation = candidate["active_evaluation"]
    return (
        float(evaluation["solvable_pass_at_1"]),
        float(evaluation["pass_at_1"]),
        float(evaluation["target_accuracy"]),
        -float(evaluation["mean_latency_regret_ms"]),
        -float(evaluation["mean_latency_ms"]),
        -float(evaluation["fallback_rate"]),
    )


def select_threshold(
    registry: Path,
    dataset: Path | None,
    regression_manifest: Path,
    thresholds: list[float],
    epochs: int,
    learning_rate: float,
    l2: float,
    label_mode: str = "rescue",
    teacher_margin: float = 0.1,
) -> dict[str, Any]:
    if not thresholds:
        raise ValueError("at least one threshold is required")
    router, active = load_active_router(registry)
    evaluated_dataset = dataset or Path(active["dataset"])
    records = read_routing_records(evaluated_dataset)
    errors = validate_routing_records(records)
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "registry": str(registry),
            "dataset": str(evaluated_dataset),
        }

    trained_head, history = train_fallback_head(
        records,
        router,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        label_mode=label_mode,
        teacher_margin=teacher_margin,
    )
    candidates = []
    for threshold in thresholds:
        head = FallbackHead(
            feature_names=trained_head.feature_names,
            weights=trained_head.weights,
            threshold=threshold,
        )
        active_evaluation = evaluate_fallback_head(records, router, head)
        regression_report = evaluate_regression_slices(registry, regression_manifest, head)
        candidates.append(
            {
                "threshold": threshold,
                "eligible": bool(regression_report["passed"]),
                "active_evaluation": active_evaluation,
                "regression_report": regression_report,
            }
        )

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(eligible, key=rank_key) if eligible else None
    return {
        "valid": True,
        "registry": str(registry),
        "base_model": active["model"],
        "base_dataset": active["dataset"],
        "evaluated_dataset": str(evaluated_dataset),
        "regression_manifest": str(regression_manifest),
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "label_mode": label_mode,
            "teacher_margin": teacher_margin if label_mode == "margin-gate" else None,
            "history": history,
        },
        "feature_count": len(trained_head.feature_names),
        "base_head": trained_head.to_dict(),
        "selection_rule": [
            "require all regression slices to pass",
            "maximize active solvable_pass_at_1",
            "maximize active pass_at_1",
            "maximize active target_accuracy",
            "minimize active mean_latency_regret_ms",
            "minimize active mean_latency_ms",
            "minimize active fallback_rate",
        ],
        "selected": selected,
        "candidates": candidates,
    }


def selected_head_payload(selection: dict[str, Any]) -> dict[str, Any]:
    selected = selection.get("selected")
    if not selected:
        raise ValueError("cannot write fallback head without a selected threshold")
    head = dict(selection["base_head"])
    head["threshold"] = selected["threshold"]
    return {
        "policy": "fallback-logit-head",
        "base_model": selection["base_model"],
        "base_dataset": selection["base_dataset"],
        "evaluated_dataset": selection["evaluated_dataset"],
        "regression_manifest": selection["regression_manifest"],
        "head": head,
        "training": selection["training"],
        "active_evaluation": selected["active_evaluation"],
        "regression_passed": selected["regression_report"]["passed"],
        "selection_rule": selection["selection_rule"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a fallback logit head for the active logits router."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--dataset", type=Path, help="Defaults to the active policy dataset.")
    parser.add_argument(
        "--regression-manifest",
        type=Path,
        default=Path("research/evals/router_regression_slices.json"),
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.5, 0.6, 0.8])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--label-mode", choices=["rescue", "margin-gate"], default="rescue")
    parser.add_argument("--teacher-margin", type=float, default=0.1)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    selection = select_threshold(
        args.registry,
        args.dataset,
        args.regression_manifest,
        thresholds=args.thresholds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        label_mode=args.label_mode,
        teacher_margin=args.teacher_margin,
    )
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not selection.get("valid", False):
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 1
    if not selection.get("selected"):
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 2

    model = selected_head_payload(selection)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selection": selection, "model": model}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
