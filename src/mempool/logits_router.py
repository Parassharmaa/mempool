from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .router_baseline import evaluate_policy
from .task_features import extract_task_features


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    offset = max(logits)
    exps = [math.exp(value - offset) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def kl_divergence(target: list[float], predicted: list[float]) -> float:
    total = 0.0
    for target_value, predicted_value in zip(target, predicted, strict=True):
        if target_value <= 0:
            continue
        total += target_value * math.log(target_value / max(predicted_value, 1e-12))
    return total


def collect_feature_names(records: list[dict[str, Any]]) -> list[str]:
    names = set()
    for record in records:
        names.update(extract_task_features(record))
    return sorted(names)


def collect_worker_ids(records: list[dict[str, Any]]) -> list[str]:
    ids = set()
    for record in records:
        for worker in record["workers"]:
            ids.add(worker["worker_id"])
    return sorted(ids)


def feature_vector(record: dict[str, Any], feature_names: list[str]) -> list[float]:
    features = extract_task_features(record)
    return [scale_feature(name, features.get(name, 0.0)) for name in feature_names]


def target_vector(record: dict[str, Any], worker_ids: list[str]) -> list[float]:
    distribution = record["target_distribution"]
    values = [float(distribution.get(worker_id, 0.0)) for worker_id in worker_ids]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(worker_ids) for _ in worker_ids]
    return [value / total for value in values]


def reward_target_vector(
    record: dict[str, Any],
    worker_ids: list[str],
    temperature: float = 0.25,
) -> list[float]:
    rewards = {
        str(worker["worker_id"]): float(worker.get("reward", 0.0) or 0.0)
        for worker in record.get("workers", [])
    }
    return softmax([rewards.get(worker_id, 0.0) / max(temperature, 1e-9) for worker_id in worker_ids])


def training_weight(record: dict[str, Any]) -> float:
    value = float(record.get("training_weight", 1.0) or 0.0)
    return max(0.0, value)


def scale_feature(name: str, value: float) -> float:
    if name.startswith("length_"):
        return value / 200.0
    if name in {"library_count", "missing_library_count", "plausibility_score"}:
        return value / 10.0
    if name == "environment_risk":
        return value / 5.0
    return value


@dataclass
class LogitsRouter:
    worker_ids: list[str]
    feature_names: list[str]
    weights: list[list[float]]

    @classmethod
    def initialize(
        cls,
        records: list[dict[str, Any]],
        initial_router: "LogitsRouter | None" = None,
    ) -> "LogitsRouter":
        worker_ids = collect_worker_ids(records)
        feature_names = collect_feature_names(records)
        weights = [[0.0 for _ in feature_names] for _ in worker_ids]
        if initial_router is not None:
            initial_worker_index = {
                worker_id: index for index, worker_id in enumerate(initial_router.worker_ids)
            }
            initial_feature_index = {
                feature_name: index for index, feature_name in enumerate(initial_router.feature_names)
            }
            for worker_index, worker_id in enumerate(worker_ids):
                source_worker_index = initial_worker_index.get(worker_id)
                if source_worker_index is None:
                    continue
                for feature_index, feature_name in enumerate(feature_names):
                    source_feature_index = initial_feature_index.get(feature_name)
                    if source_feature_index is None:
                        continue
                    weights[worker_index][feature_index] = initial_router.weights[
                        source_worker_index
                    ][source_feature_index]
        return cls(worker_ids=worker_ids, feature_names=feature_names, weights=weights)

    def logits(self, record: dict[str, Any]) -> list[float]:
        vector = feature_vector(record, self.feature_names)
        return [
            sum(weight * value for weight, value in zip(worker_weights, vector, strict=True))
            for worker_weights in self.weights
        ]

    def distribution(self, record: dict[str, Any]) -> dict[str, float]:
        probabilities = softmax(self.logits(record))
        return {
            worker_id: probability
            for worker_id, probability in zip(self.worker_ids, probabilities, strict=True)
        }

    def predict(self, record: dict[str, Any]) -> str:
        distribution = self.distribution(record)
        return max(distribution, key=distribution.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "logits-router",
            "worker_ids": self.worker_ids,
            "feature_names": self.feature_names,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LogitsRouter":
        return cls(
            worker_ids=list(payload["worker_ids"]),
            feature_names=list(payload["feature_names"]),
            weights=[list(row) for row in payload["weights"]],
        )


def train_logits_router(
    records: list[dict[str, Any]],
    epochs: int = 300,
    learning_rate: float = 0.0005,
    l2: float = 0.0001,
    target_mode: str = "distribution",
    reward_temperature: float = 0.25,
    initial_router: LogitsRouter | None = None,
) -> tuple[LogitsRouter, list[dict[str, float]]]:
    router = LogitsRouter.initialize(records, initial_router=initial_router)
    history = []
    if not records:
        return router, history

    for epoch in range(epochs):
        total_kl = 0.0
        total_weight = 0.0
        for record in records:
            weight = training_weight(record)
            if weight <= 0:
                continue
            vector = feature_vector(record, router.feature_names)
            target = (
                reward_target_vector(record, router.worker_ids, temperature=reward_temperature)
                if target_mode == "reward"
                else target_vector(record, router.worker_ids)
            )
            predicted = softmax(router.logits(record))
            total_kl += weight * kl_divergence(target, predicted)
            total_weight += weight
            for worker_index, (predicted_value, target_value) in enumerate(
                zip(predicted, target, strict=True)
            ):
                error = weight * (predicted_value - target_value)
                for feature_index, feature_value in enumerate(vector):
                    penalty = l2 * router.weights[worker_index][feature_index]
                    gradient = error * feature_value + penalty
                    router.weights[worker_index][feature_index] -= learning_rate * gradient
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "mean_kl": total_kl / total_weight if total_weight else 0.0,
                }
            )
    return router, history


def evaluate_logits_router(records: list[dict[str, Any]], router: LogitsRouter) -> dict[str, Any]:
    predictions = [router.predict(record) for record in records]
    policy_eval = evaluate_policy(records, "logits-router", predictions)
    mean_kl = 0.0
    for record in records:
        target = target_vector(record, router.worker_ids)
        predicted_distribution = router.distribution(record)
        predicted = [predicted_distribution[worker_id] for worker_id in router.worker_ids]
        mean_kl += kl_divergence(target, predicted)
    mean_kl = mean_kl / len(records) if records else 0.0
    target_latencies = []
    predicted_latencies = []
    solvable_task_count = 0
    solvable_solved = 0
    for record, prediction in zip(records, predictions, strict=True):
        target_worker = next(
            (worker for worker in record["workers"] if worker["worker_id"] == record["target_worker_id"]),
            None,
        )
        predicted_worker = next(
            (worker for worker in record["workers"] if worker["worker_id"] == prediction),
            None,
        )
        if target_worker:
            target_latencies.append(float(target_worker.get("latency_ms", 0.0) or 0.0))
        if predicted_worker:
            predicted_latencies.append(float(predicted_worker.get("latency_ms", 0.0) or 0.0))
        if any(bool(worker.get("passed")) for worker in record["workers"]):
            solvable_task_count += 1
            solvable_solved += int(bool(predicted_worker and predicted_worker.get("passed")))
    mean_target_latency = sum(target_latencies) / len(target_latencies) if target_latencies else 0.0
    mean_predicted_latency = sum(predicted_latencies) / len(predicted_latencies) if predicted_latencies else 0.0
    return {
        "policy": policy_eval.policy,
        "task_count": policy_eval.task_count,
        "matched_target": policy_eval.matched_target,
        "target_accuracy": policy_eval.target_accuracy,
        "solved": policy_eval.solved,
        "pass_at_1": policy_eval.pass_at_1,
        "mean_latency_ms": policy_eval.mean_latency_ms,
        "cost_per_solved_task": policy_eval.cost_per_solved_task,
        "mean_kl": mean_kl,
        "mean_target_latency_ms": mean_target_latency,
        "mean_latency_regret_ms": max(0.0, mean_predicted_latency - mean_target_latency),
        "solvable_task_count": solvable_task_count,
        "solvable_pass_at_1": solvable_solved / solvable_task_count if solvable_task_count else 0.0,
        "predictions": predictions,
    }


def leave_one_out_logits_evaluation(
    records: list[dict[str, Any]],
    epochs: int = 300,
    learning_rate: float = 0.0005,
    l2: float = 0.0001,
    target_mode: str = "distribution",
    reward_temperature: float = 0.25,
    initial_router: LogitsRouter | None = None,
) -> dict[str, Any]:
    if len(records) < 2:
        return {"available": False, "reason": "requires at least two records"}
    predictions = []
    examples = []
    for index, record in enumerate(records):
        training = records[:index] + records[index + 1 :]
        router, _ = train_logits_router(
            training,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            target_mode=target_mode,
            reward_temperature=reward_temperature,
            initial_router=initial_router,
        )
        prediction = router.predict(record)
        predictions.append(prediction)
        examples.append(
            {
                "task_id": record["task_id"],
                "target_worker_id": record["target_worker_id"],
                "predicted_worker_id": prediction,
            }
        )
    policy_eval = evaluate_policy(records, "logits-router-loo", predictions)
    return {
        "available": True,
        "task_count": len(records),
        "target_accuracy": policy_eval.target_accuracy,
        "pass_at_1": policy_eval.pass_at_1,
        "mean_latency_ms": policy_eval.mean_latency_ms,
        "mean_target_latency_ms": policy_eval.mean_target_latency_ms,
        "mean_latency_regret_ms": policy_eval.mean_latency_regret_ms,
        "solvable_task_count": policy_eval.solvable_task_count,
        "solvable_pass_at_1": policy_eval.solvable_pass_at_1,
        "solvable_target_accuracy": policy_eval.solvable_target_accuracy,
        "predictions": predictions,
        "examples": examples,
    }
