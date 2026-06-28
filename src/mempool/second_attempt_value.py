from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def worker_passed(worker: dict[str, Any]) -> bool:
    return float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0)) > 0.0


def worker_latency_ms(worker: dict[str, Any]) -> float:
    return float(worker.get("mean_latency_ms", worker.get("latency_ms", 0.0)) or 0.0)


def worker_by_id(record: dict[str, Any], worker_id: str) -> dict[str, Any]:
    for worker in record["workers"]:
        if str(worker["worker_id"]) == worker_id:
            return worker
    raise KeyError(worker_id)


def ranked_worker_ids(prediction: dict[str, Any]) -> list[str]:
    distribution = prediction["worker_distribution"]
    return sorted(distribution, key=distribution.get, reverse=True)


def second_attempt_value(
    record: dict[str, Any],
    prediction: dict[str, Any],
    *,
    latency_cost_per_second: float = 0.05,
) -> dict[str, Any]:
    if latency_cost_per_second < 0:
        raise ValueError("latency cost must be nonnegative")

    ranked = ranked_worker_ids(prediction)
    if len(ranked) < 2:
        return {
            "available": False,
            "reason": "fewer than two ranked workers",
            "value": 0.0,
            "label": 0.0,
        }

    top_worker = worker_by_id(record, ranked[0])
    second_worker = worker_by_id(record, ranked[1])
    top_solved = worker_passed(top_worker)
    second_solved = worker_passed(second_worker)
    solve_gain = float((not top_solved) and second_solved)
    added_latency_ms = worker_latency_ms(second_worker)
    latency_cost = (added_latency_ms / 1000.0) * latency_cost_per_second
    value = solve_gain - latency_cost
    return {
        "available": True,
        "top_worker_id": ranked[0],
        "second_worker_id": ranked[1],
        "top_solved": top_solved,
        "second_solved": second_solved,
        "solve_gain": solve_gain,
        "added_latency_ms": added_latency_ms,
        "latency_cost": latency_cost,
        "value": value,
        "label": float(value > 0.0),
    }


def evaluate_value_gated_fallback(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    latency_cost_per_second: float = 0.05,
    min_value: float = 0.0,
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length")

    solved = 0
    matched_target = 0
    solvable_task_count = 0
    solvable_solved = 0
    solvable_matched_target = 0
    fallback_opportunities = 0
    fallbacks_taken = 0
    useful_fallbacks = 0
    total_latency = 0.0
    total_target_latency = 0.0
    total_latency_regret = 0.0
    total_value = 0.0
    examples = []

    for record, prediction in zip(records, predictions, strict=True):
        ranked = ranked_worker_ids(prediction)
        if not ranked:
            continue
        top_worker = worker_by_id(record, ranked[0])
        final_worker = top_worker
        task_latency = worker_latency_ms(top_worker)
        target_worker_id = str(
            (record.get("target") or {}).get(
                "target_worker_id",
                record.get("target_worker_id"),
            )
        )
        target_worker = worker_by_id(record, target_worker_id)
        target_latency = worker_latency_ms(target_worker)
        solvable = any(worker_passed(worker) for worker in record["workers"])
        value_payload = second_attempt_value(
            record,
            prediction,
            latency_cost_per_second=latency_cost_per_second,
        )
        attempts = [
            {
                "worker_id": top_worker["worker_id"],
                "passed": worker_passed(top_worker),
                "latency_ms": worker_latency_ms(top_worker),
            }
        ]
        if (not worker_passed(top_worker)) and value_payload["available"]:
            fallback_opportunities += 1
            if float(value_payload["value"]) >= min_value:
                second_worker = worker_by_id(record, str(value_payload["second_worker_id"]))
                final_worker = second_worker
                fallbacks_taken += 1
                useful_fallbacks += int(worker_passed(second_worker))
                task_latency += worker_latency_ms(second_worker)
                attempts.append(
                    {
                        "worker_id": second_worker["worker_id"],
                        "passed": worker_passed(second_worker),
                        "latency_ms": worker_latency_ms(second_worker),
                    }
                )
        total_value += float(value_payload.get("value", 0.0))
        task_solved = worker_passed(final_worker)
        final_worker_id = str(final_worker["worker_id"])
        solved += int(task_solved)
        matched_target += int(final_worker_id == target_worker_id)
        if solvable:
            solvable_task_count += 1
            solvable_solved += int(task_solved)
            solvable_matched_target += int(final_worker_id == target_worker_id)
        total_latency += task_latency
        total_target_latency += target_latency
        total_latency_regret += max(0.0, task_latency - target_latency)
        examples.append(
            {
                "task_id": record["task_id"],
                "target_worker_id": target_worker_id,
                "attempts": attempts,
                "second_attempt_value": value_payload,
                "final_worker_id": final_worker_id,
                "solved": task_solved,
                "matched_target": final_worker_id == target_worker_id,
            }
        )

    task_count = len(records)
    return {
        "policy": "oracle-second-attempt-value-gated-fallback",
        "latency_cost_per_second": latency_cost_per_second,
        "min_value": min_value,
        "task_count": task_count,
        "matched_target": matched_target,
        "target_accuracy": matched_target / task_count if task_count else 0.0,
        "solved": solved,
        "pass_at_1": solved / task_count if task_count else 0.0,
        "solvable_task_count": solvable_task_count,
        "solvable_solved": solvable_solved,
        "solvable_pass_at_1": solvable_solved / solvable_task_count if solvable_task_count else 0.0,
        "solvable_target_accuracy": solvable_matched_target / solvable_task_count if solvable_task_count else 0.0,
        "fallback_opportunities": fallback_opportunities,
        "fallbacks_taken": fallbacks_taken,
        "useful_fallbacks": useful_fallbacks,
        "fallback_rate": fallbacks_taken / fallback_opportunities if fallback_opportunities else 0.0,
        "mean_second_attempt_value": total_value / task_count if task_count else 0.0,
        "mean_latency_ms": total_latency / task_count if task_count else 0.0,
        "mean_target_latency_ms": total_target_latency / task_count if task_count else 0.0,
        "mean_latency_regret_ms": total_latency_regret / task_count if task_count else 0.0,
        "examples": examples,
    }


@dataclass
class SecondAttemptValueHead:
    weights: list[float]
    threshold: float = 0.0

    def value(self, record: dict[str, Any], prediction: dict[str, Any]) -> float:
        features = value_features(record, prediction)
        return sum(weight * value for weight, value in zip(self.weights, features, strict=True))


def value_features(record: dict[str, Any], prediction: dict[str, Any]) -> list[float]:
    ranked = ranked_worker_ids(prediction)
    if len(ranked) < 2:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    top = worker_by_id(record, ranked[0])
    second = worker_by_id(record, ranked[1])
    top_probability = float(prediction["worker_distribution"][ranked[0]])
    second_probability = float(prediction["worker_distribution"][ranked[1]])
    margin = top_probability - second_probability
    verifier = float(prediction.get("verifier_probability", 0.0) or 0.0)
    return [
        1.0,
        float(not worker_passed(top)),
        top_probability,
        second_probability,
        margin,
        verifier - margin,
        worker_latency_ms(second) / 10000.0,
    ]


def train_second_attempt_value_head(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    latency_cost_per_second: float = 0.05,
    epochs: int = 100,
    learning_rate: float = 0.05,
    l2: float = 0.0,
    threshold: float = 0.0,
) -> tuple[SecondAttemptValueHead, list[dict[str, float]]]:
    head = SecondAttemptValueHead(weights=[0.0 for _ in value_features(records[0], predictions[0])] if records and predictions else [0.0] * 7, threshold=threshold)
    history = []
    if not records:
        return head, history
    targets = [
        second_attempt_value(record, prediction, latency_cost_per_second=latency_cost_per_second)["value"]
        for record, prediction in zip(records, predictions, strict=True)
    ]
    for epoch in range(epochs):
        total_loss = 0.0
        for record, prediction, target in zip(records, predictions, targets, strict=True):
            features = value_features(record, prediction)
            predicted = sum(weight * value for weight, value in zip(head.weights, features, strict=True))
            error = predicted - float(target)
            total_loss += error * error
            for index, feature in enumerate(features):
                head.weights[index] -= learning_rate * (error * feature + l2 * head.weights[index])
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            history.append({"epoch": float(epoch + 1), "mean_loss": total_loss / len(records)})
    return head, history


def evaluate_learned_value_head(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    head: SecondAttemptValueHead,
    *,
    latency_cost_per_second: float = 0.05,
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length")

    solved = 0
    matched_target = 0
    solvable_task_count = 0
    solvable_solved = 0
    solvable_matched_target = 0
    fallback_opportunities = 0
    fallbacks_taken = 0
    useful_fallbacks = 0
    total_latency = 0.0
    total_target_latency = 0.0
    total_latency_regret = 0.0
    total_value = 0.0
    examples = []

    for record, prediction in zip(records, predictions, strict=True):
        ranked = ranked_worker_ids(prediction)
        if not ranked:
            continue
        top_worker = worker_by_id(record, ranked[0])
        final_worker = top_worker
        task_latency = worker_latency_ms(top_worker)
        target_worker_id = str(
            (record.get("target") or {}).get(
                "target_worker_id",
                record.get("target_worker_id"),
            )
        )
        target_worker = worker_by_id(record, target_worker_id)
        target_latency = worker_latency_ms(target_worker)
        solvable = any(worker_passed(worker) for worker in record["workers"])
        learned_value = head.value(record, prediction)
        oracle_value = second_attempt_value(
            record,
            prediction,
            latency_cost_per_second=latency_cost_per_second,
        )
        attempts = [
            {
                "worker_id": top_worker["worker_id"],
                "passed": worker_passed(top_worker),
                "latency_ms": worker_latency_ms(top_worker),
            }
        ]
        if (not worker_passed(top_worker)) and oracle_value["available"]:
            fallback_opportunities += 1
            if learned_value >= head.threshold:
                second_worker = worker_by_id(record, str(oracle_value["second_worker_id"]))
                final_worker = second_worker
                fallbacks_taken += 1
                useful_fallbacks += int(worker_passed(second_worker))
                task_latency += worker_latency_ms(second_worker)
                attempts.append(
                    {
                        "worker_id": second_worker["worker_id"],
                        "passed": worker_passed(second_worker),
                        "latency_ms": worker_latency_ms(second_worker),
                    }
                )
        total_value += learned_value
        task_solved = worker_passed(final_worker)
        final_worker_id = str(final_worker["worker_id"])
        solved += int(task_solved)
        matched_target += int(final_worker_id == target_worker_id)
        if solvable:
            solvable_task_count += 1
            solvable_solved += int(task_solved)
            solvable_matched_target += int(final_worker_id == target_worker_id)
        total_latency += task_latency
        total_target_latency += target_latency
        total_latency_regret += max(0.0, task_latency - target_latency)
        examples.append(
            {
                "task_id": record["task_id"],
                "target_worker_id": target_worker_id,
                "attempts": attempts,
                "learned_second_attempt_value": learned_value,
                "oracle_second_attempt_value": oracle_value,
                "final_worker_id": final_worker_id,
                "solved": task_solved,
                "matched_target": final_worker_id == target_worker_id,
            }
        )

    task_count = len(records)
    return {
        "policy": "learned-second-attempt-value-head",
        "latency_cost_per_second": latency_cost_per_second,
        "threshold": head.threshold,
        "task_count": task_count,
        "matched_target": matched_target,
        "target_accuracy": matched_target / task_count if task_count else 0.0,
        "solved": solved,
        "pass_at_1": solved / task_count if task_count else 0.0,
        "solvable_task_count": solvable_task_count,
        "solvable_solved": solvable_solved,
        "solvable_pass_at_1": solvable_solved / solvable_task_count if solvable_task_count else 0.0,
        "solvable_target_accuracy": solvable_matched_target / solvable_task_count if solvable_task_count else 0.0,
        "fallback_opportunities": fallback_opportunities,
        "fallbacks_taken": fallbacks_taken,
        "useful_fallbacks": useful_fallbacks,
        "fallback_rate": fallbacks_taken / fallback_opportunities if fallback_opportunities else 0.0,
        "mean_learned_second_attempt_value": total_value / task_count if task_count else 0.0,
        "mean_latency_ms": total_latency / task_count if task_count else 0.0,
        "mean_target_latency_ms": total_target_latency / task_count if task_count else 0.0,
        "mean_latency_regret_ms": total_latency_regret / task_count if task_count else 0.0,
        "examples": examples,
    }


def leave_one_out_value_head_evaluation(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    latency_cost_per_second: float = 0.05,
    thresholds: list[float] | None = None,
    epochs: int = 100,
    learning_rate: float = 0.05,
    l2: float = 0.0,
) -> dict[str, Any]:
    thresholds = thresholds or [0.0]
    if len(records) < 2:
        return {"available": False, "reason": "requires at least two records", "task_count": len(records)}
    folds = []
    examples = []
    solved = 0
    for index, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
        train_records = records[:index] + records[index + 1 :]
        train_predictions = predictions[:index] + predictions[index + 1 :]
        head, _ = train_second_attempt_value_head(
            train_records,
            train_predictions,
            latency_cost_per_second=latency_cost_per_second,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            threshold=thresholds[0],
        )
        best_threshold = thresholds[0]
        head.threshold = best_threshold
        result = evaluate_learned_value_head([record], [prediction], head, latency_cost_per_second=latency_cost_per_second)
        solved += int(result["solved"])
        folds.append({"task_id": record["task_id"], "solved": bool(result["solved"]), "selected_threshold": best_threshold})
        examples.append({**result["examples"][0], "selected_threshold": best_threshold})
    task_count = len(records)
    return {
        "available": True,
        "policy": "learned-second-attempt-value-head-loo",
        "task_count": task_count,
        "pass_at_1": solved / task_count if task_count else 0.0,
        "folds": folds,
        "examples": examples,
    }
