from __future__ import annotations

from typing import Any

from .multi_head_orchestrator import _worker_lookup
from .second_attempt_value import worker_latency_ms, worker_passed


def get_target_worker_id(record: dict[str, Any]) -> str:
    if "target_worker_id" in record:
        return str(record["target_worker_id"])
    return str(record["target"]["target_worker_id"])


def latency_calibrated_worker_choice(
    record: dict[str, Any],
    prediction: dict[str, Any],
    *,
    latency_cost_per_second: float = 0.01,
    min_probability_ratio: float = 0.0,
    min_probability: float = 0.0,
) -> dict[str, Any]:
    if latency_cost_per_second < 0:
        raise ValueError("latency cost must be nonnegative")
    if min_probability_ratio < 0:
        raise ValueError("minimum probability ratio must be nonnegative")
    if min_probability < 0:
        raise ValueError("minimum probability must be nonnegative")

    distribution = prediction.get("worker_distribution") or {}
    if not distribution:
        raise ValueError("prediction missing worker_distribution")

    top_probability = max(float(value) for value in distribution.values())
    eligible = []
    for worker_id, probability_value in distribution.items():
        probability = float(probability_value)
        if probability < min_probability:
            continue
        if top_probability > 0 and probability < top_probability * min_probability_ratio:
            continue
        worker = _worker_lookup(record, str(worker_id))
        latency_ms = worker_latency_ms(worker)
        utility = probability - latency_cost_per_second * (latency_ms / 1000.0)
        eligible.append(
            {
                "worker_id": str(worker_id),
                "probability": probability,
                "latency_ms": latency_ms,
                "utility": utility,
            }
        )

    if not eligible:
        top_worker_id = max(distribution, key=distribution.get)
        worker = _worker_lookup(record, str(top_worker_id))
        latency_ms = worker_latency_ms(worker)
        eligible.append(
            {
                "worker_id": str(top_worker_id),
                "probability": float(distribution[top_worker_id]),
                "latency_ms": latency_ms,
                "utility": float(distribution[top_worker_id])
                - latency_cost_per_second * (latency_ms / 1000.0),
            }
        )

    selected = max(
        eligible,
        key=lambda item: (
            float(item["utility"]),
            float(item["probability"]),
            -float(item["latency_ms"]),
        ),
    )
    ranked = sorted(
        eligible,
        key=lambda item: (
            -float(item["utility"]),
            -float(item["probability"]),
            float(item["latency_ms"]),
        ),
    )
    return {
        "selected_worker_id": selected["worker_id"],
        "top_probability": top_probability,
        "latency_cost_per_second": latency_cost_per_second,
        "min_probability_ratio": min_probability_ratio,
        "min_probability": min_probability,
        "eligible_workers": ranked,
    }


def evaluate_latency_calibrated_predictions(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    latency_cost_per_second: float = 0.01,
    min_probability_ratio: float = 0.0,
    min_probability: float = 0.0,
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length")

    matched_target = 0
    solved = 0
    solvable_task_count = 0
    solvable_solved = 0
    solvable_matched_target = 0
    changed_from_top = 0
    total_latency = 0.0
    total_target_latency = 0.0
    total_latency_regret = 0.0
    examples = []

    for record, prediction in zip(records, predictions, strict=True):
        choice = latency_calibrated_worker_choice(
            record,
            prediction,
            latency_cost_per_second=latency_cost_per_second,
            min_probability_ratio=min_probability_ratio,
            min_probability=min_probability,
        )
        distribution = prediction["worker_distribution"]
        top_worker_id = str(prediction.get("top_worker_id") or max(distribution, key=distribution.get))
        selected_worker_id = str(choice["selected_worker_id"])
        target_worker_id = get_target_worker_id(record)
        selected_worker = _worker_lookup(record, selected_worker_id)
        target_worker = _worker_lookup(record, target_worker_id)
        selected_latency = worker_latency_ms(selected_worker)
        target_latency = worker_latency_ms(target_worker)
        selected_passed = worker_passed(selected_worker)
        solvable = any(worker_passed(worker) for worker in record["workers"])

        matched_target += int(selected_worker_id == target_worker_id)
        solved += int(selected_passed)
        if solvable:
            solvable_task_count += 1
            solvable_solved += int(selected_passed)
            solvable_matched_target += int(selected_worker_id == target_worker_id)
        changed_from_top += int(selected_worker_id != top_worker_id)
        total_latency += selected_latency
        total_target_latency += target_latency
        total_latency_regret += max(0.0, selected_latency - target_latency)
        examples.append(
            {
                "task_id": record["task_id"],
                "target_worker_id": target_worker_id,
                "top_worker_id": top_worker_id,
                "selected_worker_id": selected_worker_id,
                "selected_passed": selected_passed,
                "selected_latency_ms": selected_latency,
                "target_latency_ms": target_latency,
                "changed_from_top": selected_worker_id != top_worker_id,
                "choice": choice,
            }
        )

    task_count = len(records)
    return {
        "policy": "latency-calibrated-worker-choice",
        "latency_cost_per_second": latency_cost_per_second,
        "min_probability_ratio": min_probability_ratio,
        "min_probability": min_probability,
        "task_count": task_count,
        "matched_target": matched_target,
        "target_accuracy": matched_target / task_count if task_count else 0.0,
        "solved": solved,
        "pass_at_1": solved / task_count if task_count else 0.0,
        "solvable_task_count": solvable_task_count,
        "solvable_solved": solvable_solved,
        "solvable_pass_at_1": solvable_solved / solvable_task_count if solvable_task_count else 0.0,
        "solvable_target_accuracy": (
            solvable_matched_target / solvable_task_count if solvable_task_count else 0.0
        ),
        "changed_from_top": changed_from_top,
        "change_rate": changed_from_top / task_count if task_count else 0.0,
        "mean_latency_ms": total_latency / task_count if task_count else 0.0,
        "mean_target_latency_ms": total_target_latency / task_count if task_count else 0.0,
        "mean_latency_regret_ms": total_latency_regret / task_count if task_count else 0.0,
        "examples": examples,
    }


def rank_latency_calibrated_evaluation(
    evaluation: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    return (
        float(evaluation["solvable_pass_at_1"]),
        float(evaluation["pass_at_1"]),
        -float(evaluation["mean_latency_regret_ms"]),
        float(evaluation["target_accuracy"]),
        -float(evaluation["change_rate"]),
    )
