from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logits_router import kl_divergence, softmax
from .orchestrator_contract import WORKFLOW_KINDS, sigmoid


def read_substrate(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_substrate_records(records: list[dict[str, Any]]) -> list[str]:
    errors = []
    for index, record in enumerate(records):
        target = record.get("target")
        if not isinstance(target, dict):
            errors.append(f"record {index}: missing target")
            continue
        for field in [
            "worker_distribution",
            "workflow_distribution",
            "workflow_kind",
            "verifier_probability",
            "abstain_probability",
        ]:
            if field not in target:
                errors.append(f"record {index}: target missing {field}")
        if not isinstance(record.get("dense_features"), dict):
            errors.append(f"record {index}: dense_features must be an object")
        workers = record.get("workers")
        if not isinstance(workers, list) or not workers:
            errors.append(f"record {index}: workers must be a non-empty list")
    return errors


def collect_feature_names(records: list[dict[str, Any]]) -> list[str]:
    names = set()
    for record in records:
        names.update((record.get("dense_features") or {}).keys())
    return sorted(str(name) for name in names)


def collect_worker_ids(records: list[dict[str, Any]]) -> list[str]:
    ids = set()
    for record in records:
        ids.update((record.get("target") or {}).get("worker_distribution", {}).keys())
    return sorted(str(worker_id) for worker_id in ids)


def feature_scale(name: str, value: float) -> float:
    if name == "length_chars":
        return value / 2000.0
    if name == "length_tokens":
        return value / 400.0
    if name in {"library_count", "missing_library_count", "plausibility_score"}:
        return value / 10.0
    if name == "environment_risk":
        return value / 5.0
    return value


def feature_vector(record: dict[str, Any], feature_names: list[str]) -> list[float]:
    features = record.get("dense_features") or {}
    return [
        feature_scale(name, float(features.get(name, 0.0) or 0.0))
        for name in feature_names
    ]


def distribution_vector(distribution: dict[str, float], labels: list[str]) -> list[float]:
    values = [float(distribution.get(label, 0.0) or 0.0) for label in labels]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(labels) for _ in labels]
    return [value / total for value in values]


def binary_cross_entropy(target: float, predicted: float) -> float:
    clipped = min(max(predicted, 1e-12), 1.0 - 1e-12)
    return -(target * math.log(clipped) + (1.0 - target) * math.log(1.0 - clipped))


@dataclass
class MultiHeadOrchestrator:
    worker_ids: list[str]
    workflow_labels: list[str]
    feature_names: list[str]
    worker_weights: list[list[float]]
    workflow_weights: list[list[float]]
    verifier_weights: list[float]
    abstain_weights: list[float]

    @classmethod
    def initialize(cls, records: list[dict[str, Any]]) -> "MultiHeadOrchestrator":
        feature_names = collect_feature_names(records)
        worker_ids = collect_worker_ids(records)
        return cls(
            worker_ids=worker_ids,
            workflow_labels=list(WORKFLOW_KINDS),
            feature_names=feature_names,
            worker_weights=[[0.0 for _ in feature_names] for _ in worker_ids],
            workflow_weights=[[0.0 for _ in feature_names] for _ in WORKFLOW_KINDS],
            verifier_weights=[0.0 for _ in feature_names],
            abstain_weights=[0.0 for _ in feature_names],
        )

    def _linear(self, weights: list[float], vector: list[float]) -> float:
        return sum(weight * value for weight, value in zip(weights, vector, strict=True))

    def worker_logits(self, record: dict[str, Any]) -> list[float]:
        vector = feature_vector(record, self.feature_names)
        return [self._linear(weights, vector) for weights in self.worker_weights]

    def workflow_logits(self, record: dict[str, Any]) -> list[float]:
        vector = feature_vector(record, self.feature_names)
        return [self._linear(weights, vector) for weights in self.workflow_weights]

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        vector = feature_vector(record, self.feature_names)
        worker_probabilities = softmax(
            [self._linear(weights, vector) for weights in self.worker_weights]
        )
        workflow_probabilities = softmax(
            [self._linear(weights, vector) for weights in self.workflow_weights]
        )
        worker_distribution = {
            worker_id: probability
            for worker_id, probability in zip(self.worker_ids, worker_probabilities, strict=True)
        }
        workflow_distribution = {
            label: probability
            for label, probability in zip(self.workflow_labels, workflow_probabilities, strict=True)
        }
        return {
            "worker_distribution": worker_distribution,
            "target_worker_id": max(worker_distribution, key=worker_distribution.get),
            "workflow_distribution": workflow_distribution,
            "workflow_kind": max(workflow_distribution, key=workflow_distribution.get),
            "verifier_probability": sigmoid(self._linear(self.verifier_weights, vector)),
            "abstain_probability": sigmoid(self._linear(self.abstain_weights, vector)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "linear-multi-head-orchestrator",
            "worker_ids": self.worker_ids,
            "workflow_labels": self.workflow_labels,
            "feature_names": self.feature_names,
            "worker_weights": self.worker_weights,
            "workflow_weights": self.workflow_weights,
            "verifier_weights": self.verifier_weights,
            "abstain_weights": self.abstain_weights,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MultiHeadOrchestrator":
        return cls(
            worker_ids=list(payload["worker_ids"]),
            workflow_labels=list(payload["workflow_labels"]),
            feature_names=list(payload["feature_names"]),
            worker_weights=[list(row) for row in payload["worker_weights"]],
            workflow_weights=[list(row) for row in payload["workflow_weights"]],
            verifier_weights=list(payload["verifier_weights"]),
            abstain_weights=list(payload["abstain_weights"]),
        )


def target_worker_vector(record: dict[str, Any], worker_ids: list[str]) -> list[float]:
    return distribution_vector(record["target"]["worker_distribution"], worker_ids)


def target_workflow_vector(record: dict[str, Any], workflow_labels: list[str]) -> list[float]:
    return distribution_vector(record["target"]["workflow_distribution"], workflow_labels)


def train_multi_head_orchestrator(
    records: list[dict[str, Any]],
    *,
    epochs: int = 300,
    learning_rate: float = 0.0005,
    l2: float = 0.0001,
    latency_regret_weight: float = 0.0,
) -> tuple[MultiHeadOrchestrator, list[dict[str, float]]]:
    if latency_regret_weight < 0:
        raise ValueError("latency_regret_weight must be nonnegative")
    model = MultiHeadOrchestrator.initialize(records)
    history = []
    if not records:
        return model, history

    for epoch in range(epochs):
        total_loss = 0.0
        for record in records:
            vector = feature_vector(record, model.feature_names)
            worker_target = target_worker_vector(record, model.worker_ids)
            workflow_target = target_workflow_vector(record, model.workflow_labels)
            worker_predicted = softmax(
                [model._linear(weights, vector) for weights in model.worker_weights]
            )
            workflow_predicted = softmax(
                [model._linear(weights, vector) for weights in model.workflow_weights]
            )
            verifier_target = float(record["target"]["verifier_probability"])
            abstain_target = float(record["target"]["abstain_probability"])
            verifier_predicted = sigmoid(model._linear(model.verifier_weights, vector))
            abstain_predicted = sigmoid(model._linear(model.abstain_weights, vector))

            total_loss += kl_divergence(worker_target, worker_predicted)
            total_loss += kl_divergence(workflow_target, workflow_predicted)
            total_loss += binary_cross_entropy(verifier_target, verifier_predicted)
            total_loss += binary_cross_entropy(abstain_target, abstain_predicted)

            for row_index, (predicted, target) in enumerate(zip(worker_predicted, worker_target, strict=True)):
                error = predicted - target
                for feature_index, feature_value in enumerate(vector):
                    penalty = l2 * model.worker_weights[row_index][feature_index]
                    model.worker_weights[row_index][feature_index] -= learning_rate * (
                        error * feature_value + penalty
                    )
            for row_index, (predicted, target) in enumerate(zip(workflow_predicted, workflow_target, strict=True)):
                error = predicted - target
                for feature_index, feature_value in enumerate(vector):
                    penalty = l2 * model.workflow_weights[row_index][feature_index]
                    model.workflow_weights[row_index][feature_index] -= learning_rate * (
                        error * feature_value + penalty
                    )
            for weights, predicted, target in [
                (model.verifier_weights, verifier_predicted, verifier_target),
                (model.abstain_weights, abstain_predicted, abstain_target),
            ]:
                error = predicted - target
                for feature_index, feature_value in enumerate(vector):
                    penalty = l2 * weights[feature_index]
                    weights[feature_index] -= learning_rate * (error * feature_value + penalty)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            history.append({"epoch": float(epoch + 1), "mean_loss": total_loss / len(records)})
    return model, history


def _worker_lookup(record: dict[str, Any], worker_id: str) -> dict[str, Any]:
    for worker in record["workers"]:
        if worker["worker_id"] == worker_id:
            return worker
    raise KeyError(worker_id)


def evaluate_multi_head_orchestrator(
    records: list[dict[str, Any]],
    model: MultiHeadOrchestrator,
) -> dict[str, Any]:
    if not records:
        return {"policy": "linear-multi-head-orchestrator", "task_count": 0}

    matched_worker = 0
    matched_workflow = 0
    solved = 0
    solvable_count = 0
    solvable_solved = 0
    solvable_matched_worker = 0
    abstain_matches = 0
    verifier_bce = 0.0
    abstain_bce = 0.0
    worker_kl = 0.0
    workflow_kl = 0.0
    total_latency = 0.0
    total_target_latency = 0.0
    total_latency_regret = 0.0
    predictions = []

    for record in records:
        prediction = model.predict(record)
        predicted_worker_id = prediction["target_worker_id"]
        predicted_workflow = prediction["workflow_kind"]
        target_worker_id = record["target"]["target_worker_id"]
        target_workflow = record["target"]["workflow_kind"]
        chosen_worker = _worker_lookup(record, predicted_worker_id)
        target_worker = _worker_lookup(record, target_worker_id)
        solvable = any(float(worker.get("pass_rate", 0.0)) > 0.0 for worker in record["workers"])
        matched_worker += int(predicted_worker_id == target_worker_id)
        matched_workflow += int(predicted_workflow == target_workflow)
        solved += int(float(chosen_worker.get("pass_rate", 0.0)) > 0.0)
        if solvable:
            solvable_count += 1
            solvable_solved += int(float(chosen_worker.get("pass_rate", 0.0)) > 0.0)
            solvable_matched_worker += int(predicted_worker_id == target_worker_id)
        abstain_target = float(record["target"]["abstain_probability"])
        abstain_prediction = float(prediction["abstain_probability"])
        verifier_target = float(record["target"]["verifier_probability"])
        verifier_prediction = float(prediction["verifier_probability"])
        abstain_matches += int((abstain_prediction >= 0.5) == (abstain_target >= 0.5))
        verifier_bce += binary_cross_entropy(verifier_target, verifier_prediction)
        abstain_bce += binary_cross_entropy(abstain_target, abstain_prediction)
        worker_kl += kl_divergence(
            target_worker_vector(record, model.worker_ids),
            [prediction["worker_distribution"][worker_id] for worker_id in model.worker_ids],
        )
        workflow_kl += kl_divergence(
            target_workflow_vector(record, model.workflow_labels),
            [prediction["workflow_distribution"][label] for label in model.workflow_labels],
        )
        chosen_latency = float(chosen_worker.get("mean_latency_ms", 0.0))
        target_latency = float(target_worker.get("mean_latency_ms", 0.0))
        total_latency += chosen_latency
        total_target_latency += target_latency
        total_latency_regret += max(0.0, chosen_latency - target_latency)
        predictions.append(
            {
                "task_id": record["task_id"],
                "target_worker_id": target_worker_id,
                "predicted_worker_id": predicted_worker_id,
                "target_workflow_kind": target_workflow,
                "predicted_workflow_kind": predicted_workflow,
                "verifier_probability": verifier_prediction,
                "abstain_probability": abstain_prediction,
            }
        )

    task_count = len(records)
    return {
        "policy": "linear-multi-head-orchestrator",
        "task_count": task_count,
        "matched_target": matched_worker,
        "target_accuracy": matched_worker / task_count,
        "workflow_accuracy": matched_workflow / task_count,
        "abstain_accuracy": abstain_matches / task_count,
        "solved": solved,
        "pass_at_1": solved / task_count,
        "solvable_task_count": solvable_count,
        "solvable_solved": solvable_solved,
        "solvable_pass_at_1": solvable_solved / solvable_count if solvable_count else 0.0,
        "solvable_target_accuracy": solvable_matched_worker / solvable_count if solvable_count else 0.0,
        "mean_latency_ms": total_latency / task_count,
        "mean_target_latency_ms": total_target_latency / task_count,
        "mean_latency_regret_ms": total_latency_regret / task_count,
        "mean_worker_kl": worker_kl / task_count,
        "mean_workflow_kl": workflow_kl / task_count,
        "mean_verifier_bce": verifier_bce / task_count,
        "mean_abstain_bce": abstain_bce / task_count,
        "predictions": predictions,
    }


def leave_one_out_multi_head_evaluation(
    records: list[dict[str, Any]],
    *,
    epochs: int = 300,
    learning_rate: float = 0.0005,
    l2: float = 0.0001,
) -> dict[str, Any]:
    if len(records) < 2:
        return {
            "available": False,
            "reason": "leave-one-out requires at least two records",
            "task_count": len(records),
        }
    predictions = []
    for index, record in enumerate(records):
        train_records = records[:index] + records[index + 1 :]
        model, _ = train_multi_head_orchestrator(
            train_records,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        predictions.append(model.predict(record))

    class FixedModel:
        def __init__(self, rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> None:
            self.by_task = {row["task_id"]: output for row, output in zip(rows, outputs, strict=True)}
            first = outputs[0]
            self.worker_ids = list(first["worker_distribution"])
            self.workflow_labels = list(first["workflow_distribution"])

        def predict(self, record: dict[str, Any]) -> dict[str, Any]:
            return self.by_task[record["task_id"]]

    result = evaluate_multi_head_orchestrator(records, FixedModel(records, predictions))  # type: ignore[arg-type]
    result["available"] = True
    result["policy"] = "linear-multi-head-orchestrator-loo"
    return result


def latency_regret_vector(record: dict[str, Any], worker_ids: list[str]) -> list[float]:
    target_worker = _worker_lookup(record, record["target"]["target_worker_id"])
    target_latency = float(target_worker.get("mean_latency_ms", 0.0) or 0.0)
    return [
        max(0.0, float(_worker_lookup(record, worker_id).get("mean_latency_ms", 0.0) or 0.0) - target_latency)
        for worker_id in worker_ids
    ]


def target_balance_weights(records: list[dict[str, Any]], power: float = 0.5) -> dict[str, float]:
    counts: dict[str, int] = {}
    for record in records:
        target = str(record["target"]["target_worker_id"])
        counts[target] = counts.get(target, 0) + 1
    raw = {}
    for record in records:
        count = counts[str(record["target"]["target_worker_id"])]
        raw[record["task_id"]] = 1.0 / (count ** power)
    mean = sum(raw.values()) / len(raw) if raw else 1.0
    return {task_id: value / mean for task_id, value in raw.items()}


def evaluate_multi_head_fallback_predictions(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    max_attempts: int = 2,
    max_first_second_margin: float = 0.1,
    min_verifier_probability: float = 0.5,
) -> dict[str, Any]:
    solved = 0
    matched_target = 0
    solvable_task_count = 0
    solvable_solved = 0
    solvable_matched_target = 0
    fallbacks_taken = 0
    fallback_opportunities = 0
    verifier_blocks = 0
    total_latency = 0.0
    total_target_latency = 0.0
    total_latency_regret = 0.0
    examples = []
    for record, prediction in zip(records, predictions, strict=True):
        ranked = sorted(prediction["worker_distribution"], key=prediction["worker_distribution"].get, reverse=True)
        top_worker = _worker_lookup(record, ranked[0])
        final_worker = top_worker
        task_latency = float(top_worker.get("mean_latency_ms", 0.0) or 0.0)
        attempts = [{"worker_id": ranked[0], "passed": float(top_worker.get("pass_rate", 0.0) or 0.0) > 0.0}]
        if len(ranked) > 1 and max_attempts > 1 and not attempts[0]["passed"]:
            fallback_opportunities += 1
            margin = float(prediction["worker_distribution"][ranked[0]]) - float(prediction["worker_distribution"][ranked[1]])
            verifier_probability = float(prediction.get("verifier_probability", 0.0) or 0.0)
            if margin <= max_first_second_margin and verifier_probability >= min_verifier_probability:
                second_worker = _worker_lookup(record, ranked[1])
                final_worker = second_worker
                task_latency += float(second_worker.get("mean_latency_ms", 0.0) or 0.0)
                fallbacks_taken += 1
                attempts.append({"worker_id": ranked[1], "passed": float(second_worker.get("pass_rate", 0.0) or 0.0) > 0.0})
            elif verifier_probability < min_verifier_probability:
                verifier_blocks += 1
        task_solved = float(final_worker.get("pass_rate", 0.0) or 0.0) > 0.0
        solved += int(task_solved)
        target_worker_id = str(record["target"]["target_worker_id"])
        target_worker = _worker_lookup(record, target_worker_id)
        target_latency = float(target_worker.get("mean_latency_ms", 0.0) or 0.0)
        total_latency += task_latency
        total_target_latency += target_latency
        total_latency_regret += max(0.0, task_latency - target_latency)
        matched_target += int(str(final_worker["worker_id"]) == target_worker_id)
        solvable = any(float(worker.get("pass_rate", 0.0) or 0.0) > 0.0 for worker in record["workers"])
        if solvable:
            solvable_task_count += 1
            solvable_solved += int(task_solved)
            solvable_matched_target += int(str(final_worker["worker_id"]) == target_worker_id)
        examples.append(
            {
                "task_id": record["task_id"],
                "attempts": attempts,
                "final_worker_id": final_worker["worker_id"],
                "solved": task_solved,
            }
        )
    return {
        "policy": "multi-head-fallback-gate",
        "task_count": len(records),
        "matched_target": matched_target,
        "target_accuracy": matched_target / len(records) if records else 0.0,
        "solved": solved,
        "pass_at_1": solved / len(records) if records else 0.0,
        "solvable_task_count": solvable_task_count,
        "solvable_solved": solvable_solved,
        "solvable_pass_at_1": solvable_solved / solvable_task_count if solvable_task_count else 0.0,
        "solvable_target_accuracy": solvable_matched_target / solvable_task_count if solvable_task_count else 0.0,
        "mean_latency_ms": total_latency / len(records) if records else 0.0,
        "mean_target_latency_ms": total_target_latency / len(records) if records else 0.0,
        "mean_latency_regret_ms": total_latency_regret / len(records) if records else 0.0,
        "fallback_opportunities": fallback_opportunities,
        "fallbacks_taken": fallbacks_taken,
        "fallback_rate": fallbacks_taken / fallback_opportunities if fallback_opportunities else 0.0,
        "verifier_blocks": verifier_blocks,
        "examples": examples,
    }
