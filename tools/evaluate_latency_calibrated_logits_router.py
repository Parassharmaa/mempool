from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.latency_calibrated_router import (
    evaluate_latency_calibrated_predictions,
    latency_calibrated_worker_choice,
    rank_latency_calibrated_evaluation,
)
from mempool.latency_safe_probe import probe_policy_prediction
from mempool.logits_router import LogitsRouter, evaluate_logits_router
from mempool.routing_dataset import read_routing_records, validate_routing_records


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def concise_evaluation(evaluation: dict) -> dict:
    return {
        key: evaluation.get(key)
        for key in [
            "policy",
            "latency_cost_per_second",
            "min_probability_ratio",
            "min_probability",
            "task_count",
            "target_accuracy",
            "pass_at_1",
            "solvable_pass_at_1",
            "mean_latency_ms",
            "mean_target_latency_ms",
            "mean_latency_regret_ms",
            "changed_from_top",
            "change_rate",
            "conditional_calibration",
            "calibrated_task_count",
            "source_latency_cost_per_second",
            "source_min_probability_ratio",
            "source_min_probability",
        ]
    }


def evaluate_grid(
    *,
    records: list[dict],
    router: LogitsRouter,
    latency_costs: list[float],
    min_probability_ratios: list[float],
    min_probabilities: list[float],
) -> dict:
    predictions = [{"worker_distribution": router.distribution(record)} for record in records]
    candidates = []
    for latency_cost in latency_costs:
        for min_probability_ratio in min_probability_ratios:
            for min_probability in min_probabilities:
                candidates.append(
                    evaluate_latency_calibrated_predictions(
                        records,
                        predictions,
                        latency_cost_per_second=latency_cost,
                        min_probability_ratio=min_probability_ratio,
                        min_probability=min_probability,
                    )
                )
    return max(candidates, key=rank_latency_calibrated_evaluation), candidates


def top_worker_id(prediction: dict) -> str:
    distribution = prediction["worker_distribution"]
    return str(max(distribution, key=distribution.get))


def one_hot_prediction(
    worker_ids: list[str],
    selected_worker_id: str,
    *,
    top_worker_id: str | None = None,
) -> dict:
    return {
        "worker_distribution": {
            worker_id: 1.0 if worker_id == selected_worker_id else 0.0
            for worker_id in worker_ids
        },
        "top_worker_id": top_worker_id or selected_worker_id,
    }


def conditionally_calibrated_predictions(
    *,
    records: list[dict],
    router: LogitsRouter,
    calibrate_task_ids: set[str],
    latency_cost_per_second: float,
    min_probability_ratio: float,
    min_probability: float,
) -> list[dict]:
    worker_ids = router.worker_ids
    predictions = []
    for record in records:
        prediction = {"worker_distribution": router.distribution(record)}
        if str(record["task_id"]) in calibrate_task_ids:
            choice = latency_calibrated_worker_choice(
                record,
                prediction,
                latency_cost_per_second=latency_cost_per_second,
                min_probability_ratio=min_probability_ratio,
                min_probability=min_probability,
            )
            selected_worker_id = str(choice["selected_worker_id"])
        else:
            selected_worker_id = top_worker_id(prediction)
        predictions.append(
            one_hot_prediction(
                worker_ids,
                selected_worker_id,
                top_worker_id=top_worker_id(prediction),
            )
        )
    return predictions


def probe_gated_calibrate_task_ids(
    *,
    records: list[dict],
    probe_worker_ids: list[str],
    probe_mode: str = "all",
    min_pass_rate: float = 1.0,
) -> set[str]:
    return {
        str(record["task_id"])
        for record in records
        if probe_policy_prediction(
            record,
            probe_worker_ids,
            mode=probe_mode,
            min_pass_rate=min_pass_rate,
        )
    }


def evaluate_conditional_grid(
    *,
    records: list[dict],
    router: LogitsRouter,
    calibrate_task_ids: set[str],
    latency_costs: list[float],
    min_probability_ratios: list[float],
    min_probabilities: list[float],
) -> tuple[dict, list[dict]]:
    candidates = []
    for latency_cost in latency_costs:
        for min_probability_ratio in min_probability_ratios:
            for min_probability in min_probabilities:
                predictions = conditionally_calibrated_predictions(
                    records=records,
                    router=router,
                    calibrate_task_ids=calibrate_task_ids,
                    latency_cost_per_second=latency_cost,
                    min_probability_ratio=min_probability_ratio,
                    min_probability=min_probability,
                )
                evaluation = evaluate_latency_calibrated_predictions(
                    records,
                    predictions,
                    latency_cost_per_second=0.0,
                    min_probability_ratio=0.0,
                    min_probability=0.0,
                )
                evaluation["conditional_calibration"] = True
                evaluation["calibrated_task_count"] = sum(
                    1 for record in records if str(record["task_id"]) in calibrate_task_ids
                )
                evaluation["source_latency_cost_per_second"] = latency_cost
                evaluation["source_min_probability_ratio"] = min_probability_ratio
                evaluation["source_min_probability"] = min_probability
                candidates.append(evaluation)
    return max(candidates, key=rank_latency_calibrated_evaluation), candidates


def read_task_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate latency-calibrated worker choice on a logits-router dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--latency-costs", type=float, nargs="+", default=[0.0, 0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--min-probability-ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--min-probabilities", type=float, nargs="+", default=[0.0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument(
        "--calibrate-task-ids",
        type=Path,
        help="If set, apply latency calibration only to listed task ids and keep top logits choice elsewhere.",
    )
    parser.add_argument(
        "--probe-worker-ids",
        nargs="+",
        help="If set, derive the conditional calibration set from probe-worker pass agreement.",
    )
    parser.add_argument("--probe-mode", choices=["all", "any"], default="all")
    parser.add_argument("--probe-min-pass-rate", type=float, default=1.0)
    args = parser.parse_args()

    records = read_routing_records(args.dataset)
    errors = validate_routing_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    model_payload = read_json(args.model)
    router = LogitsRouter.from_dict(model_payload["router"])
    base = evaluate_logits_router(records, router)
    calibrate_task_ids = read_task_ids(args.calibrate_task_ids)
    probe_gate = None
    if args.probe_worker_ids:
        calibrate_task_ids = probe_gated_calibrate_task_ids(
            records=records,
            probe_worker_ids=args.probe_worker_ids,
            probe_mode=args.probe_mode,
            min_pass_rate=args.probe_min_pass_rate,
        )
        probe_gate = {
            "probe_worker_ids": args.probe_worker_ids,
            "probe_mode": args.probe_mode,
            "probe_min_pass_rate": args.probe_min_pass_rate,
            "calibrated_task_ids": sorted(calibrate_task_ids),
            "calibrated_task_count": len(calibrate_task_ids),
        }
    if calibrate_task_ids:
        selected, candidates = evaluate_conditional_grid(
            records=records,
            router=router,
            calibrate_task_ids=calibrate_task_ids,
            latency_costs=args.latency_costs,
            min_probability_ratios=args.min_probability_ratios,
            min_probabilities=args.min_probabilities,
        )
    else:
        selected, candidates = evaluate_grid(
            records=records,
            router=router,
            latency_costs=args.latency_costs,
            min_probability_ratios=args.min_probability_ratios,
            min_probabilities=args.min_probabilities,
        )
    payload = {
        "valid": True,
        "dataset": str(args.dataset),
        "model": str(args.model),
        "calibrate_task_ids": str(args.calibrate_task_ids) if args.calibrate_task_ids else None,
        "probe_gate": probe_gate,
        "base_evaluation": base,
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.model_output:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        calibration_payload = {
            "policy": selected["policy"],
            "latency_cost_per_second": selected.get(
                "source_latency_cost_per_second",
                selected["latency_cost_per_second"],
            ),
            "min_probability_ratio": selected.get(
                "source_min_probability_ratio",
                selected["min_probability_ratio"],
            ),
            "min_probability": selected.get(
                "source_min_probability",
                selected["min_probability"],
            ),
            "conditional_calibration": bool(selected.get("conditional_calibration")),
            "calibrated_task_count": selected.get("calibrated_task_count"),
        }
        args.model_output.write_text(
            json.dumps(
                {
                    "model_type": "latency-calibrated-logits-router",
                    "base_model": str(args.model),
                    "dataset": str(args.dataset),
                    "calibrate_task_ids": str(args.calibrate_task_ids) if args.calibrate_task_ids else None,
                    "probe_gate": probe_gate,
                    "calibration": calibration_payload,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "valid": True,
                "dataset": str(args.dataset),
                "model": str(args.model),
                "base": concise_evaluation(base),
                "selected": concise_evaluation(selected),
                "probe_gate": probe_gate,
                "candidate_count": len(candidates),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
