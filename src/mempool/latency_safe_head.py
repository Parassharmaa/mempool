from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .logits_router import LogitsRouter, scale_feature
from .outcome_mining import is_broad_pass_latency_row
from .task_features import extract_task_features, feature_safe_name


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def binary_cross_entropy(target: float, predicted: float) -> float:
    clipped = min(max(predicted, 1e-12), 1.0 - 1e-12)
    return -(target * math.log(clipped) + (1.0 - target) * math.log(1.0 - clipped))


def entropy(values: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in values)


def latency_safe_label(record: dict[str, Any], min_pass_rate: float = 1.0) -> float:
    return float(is_broad_pass_latency_row(record, min_pass_rate=min_pass_rate))


def worker_reliability_context(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for record in records:
        for worker in record.get("workers", []):
            worker_id = str(worker.get("worker_id", ""))
            if not worker_id:
                continue
            item = stats.setdefault(
                worker_id,
                {
                    "attempts": 0.0,
                    "passed_attempts": 0.0,
                    "stable_rows": 0.0,
                    "rows": 0.0,
                },
            )
            attempts = float(worker.get("attempts", 1.0) or 1.0)
            pass_rate = float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0))
            item["attempts"] += attempts
            item["passed_attempts"] += pass_rate * attempts
            item["rows"] += 1.0
            item["stable_rows"] += float(pass_rate >= 1.0)
    return {
        worker_id: {
            "pass_rate": item["passed_attempts"] / item["attempts"] if item["attempts"] else 0.0,
            "stable_rate": item["stable_rows"] / item["rows"] if item["rows"] else 0.0,
            "row_count": item["rows"],
        }
        for worker_id, item in stats.items()
    }


def reliability_feature_block(
    worker_ids: list[str],
    reliability_context: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not worker_ids:
        return {}
    pass_rates = [
        float(reliability_context.get(worker_id, {}).get("pass_rate", 0.0))
        for worker_id in worker_ids
    ]
    stable_rates = [
        float(reliability_context.get(worker_id, {}).get("stable_rate", 0.0))
        for worker_id in worker_ids
    ]
    return {
        "reliability_top_pass_rate": pass_rates[0],
        "reliability_top_stable_rate": stable_rates[0],
        "reliability_min_pass_rate": min(pass_rates),
        "reliability_mean_pass_rate": sum(pass_rates) / len(pass_rates),
        "reliability_min_stable_rate": min(stable_rates),
        "reliability_mean_stable_rate": sum(stable_rates) / len(stable_rates),
    }


def latency_safe_features(
    record: dict[str, Any],
    router: LogitsRouter | None = None,
    reliability_context: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    features = {"bias": 1.0}
    features.update(extract_task_features(record))
    if router is None:
        if reliability_context:
            features.update(
                reliability_feature_block(
                    [str(worker.get("worker_id", "")) for worker in record.get("workers", [])],
                    reliability_context,
                )
            )
        return features

    distribution = router.distribution(record)
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    probabilities = [probability for _, probability in ranked]
    top_worker_id = ranked[0][0] if ranked else ""
    second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    top_probability = ranked[0][1] if ranked else 0.0
    features.update(
        {
            "top_probability": top_probability,
            "second_probability": second_probability,
            "first_second_margin": top_probability - second_probability,
            "router_entropy": entropy(probabilities),
        }
    )
    if top_worker_id:
        features[f"top_worker_{feature_safe_name(top_worker_id)}"] = 1.0
    if reliability_context:
        features.update(
            reliability_feature_block(
                [worker_id for worker_id, _ in ranked],
                reliability_context,
            )
        )
    return features


def collect_latency_safe_feature_names(
    records: list[dict[str, Any]],
    router: LogitsRouter | None = None,
    reliability_context: dict[str, dict[str, float]] | None = None,
) -> list[str]:
    names = set()
    for record in records:
        names.update(
            latency_safe_features(
                record,
                router=router,
                reliability_context=reliability_context,
            )
        )
    return sorted(names)


def latency_safe_feature_vector(
    record: dict[str, Any],
    feature_names: list[str],
    router: LogitsRouter | None = None,
    reliability_context: dict[str, dict[str, float]] | None = None,
) -> list[float]:
    features = latency_safe_features(
        record,
        router=router,
        reliability_context=reliability_context,
    )
    return [scale_feature(name, features.get(name, 0.0)) for name in feature_names]


@dataclass
class LatencySafeHead:
    feature_names: list[str]
    weights: list[float]
    threshold: float = 0.5

    def logit(
        self,
        record: dict[str, Any],
        router: LogitsRouter | None = None,
        reliability_context: dict[str, dict[str, float]] | None = None,
    ) -> float:
        vector = latency_safe_feature_vector(
            record,
            self.feature_names,
            router=router,
            reliability_context=reliability_context,
        )
        return sum(weight * value for weight, value in zip(self.weights, vector, strict=True))

    def probability(
        self,
        record: dict[str, Any],
        router: LogitsRouter | None = None,
        reliability_context: dict[str, dict[str, float]] | None = None,
    ) -> float:
        return sigmoid(self.logit(record, router=router, reliability_context=reliability_context))

    def is_latency_safe(self, record: dict[str, Any], router: LogitsRouter | None = None) -> bool:
        return self.probability(record, router=router) >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "latency-safe-logit-head",
            "feature_names": self.feature_names,
            "weights": self.weights,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LatencySafeHead":
        return cls(
            feature_names=list(payload["feature_names"]),
            weights=[float(value) for value in payload["weights"]],
            threshold=float(payload.get("threshold", 0.5)),
        )


def train_latency_safe_head(
    records: list[dict[str, Any]],
    router: LogitsRouter | None = None,
    *,
    epochs: int = 300,
    learning_rate: float = 0.01,
    l2: float = 0.0001,
    threshold: float = 0.5,
    positive_weight: float = 1.0,
    use_reliability_features: bool = False,
) -> tuple[LatencySafeHead, list[dict[str, float]]]:
    reliability_context = worker_reliability_context(records) if use_reliability_features else None
    feature_names = collect_latency_safe_feature_names(
        records,
        router=router,
        reliability_context=reliability_context,
    )
    head = LatencySafeHead(
        feature_names=feature_names,
        weights=[0.0 for _ in feature_names],
        threshold=threshold,
    )
    history = []
    if not records:
        return head, history

    for epoch in range(epochs):
        total_loss = 0.0
        total_weight = 0.0
        for record in records:
            vector = latency_safe_feature_vector(
                record,
                feature_names,
                router=router,
                reliability_context=reliability_context,
            )
            target = latency_safe_label(record)
            weight = positive_weight if target >= 0.5 else 1.0
            predicted = sigmoid(
                sum(value * feature for value, feature in zip(head.weights, vector, strict=True))
            )
            total_loss += weight * binary_cross_entropy(target, predicted)
            total_weight += weight
            error = weight * (predicted - target)
            for index, feature_value in enumerate(vector):
                penalty = l2 * head.weights[index]
                head.weights[index] -= learning_rate * (error * feature_value + penalty)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "mean_loss": total_loss / total_weight if total_weight else 0.0,
                }
            )
    return head, history


def evaluate_latency_safe_head(
    records: list[dict[str, Any]],
    head: LatencySafeHead,
    router: LogitsRouter | None = None,
    reliability_context: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    true_positive = false_positive = true_negative = false_negative = 0
    examples = []
    for record in records:
        label = latency_safe_label(record)
        probability = head.probability(
            record,
            router=router,
            reliability_context=reliability_context,
        )
        predicted = float(probability >= head.threshold)
        true_positive += int(predicted == 1.0 and label == 1.0)
        false_positive += int(predicted == 1.0 and label == 0.0)
        true_negative += int(predicted == 0.0 and label == 0.0)
        false_negative += int(predicted == 0.0 and label == 1.0)
        examples.append(
            {
                "task_id": record["task_id"],
                "label": label,
                "probability": probability,
                "predicted": predicted,
            }
        )
    total = len(records)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "policy": "latency-safe-logit-head",
        "task_count": total,
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "positive_count": true_positive + false_negative,
        "predicted_positive_count": true_positive + false_positive,
        "examples": examples,
    }


def leave_one_out_latency_safe_evaluation(
    records: list[dict[str, Any]],
    router: LogitsRouter | None = None,
    *,
    epochs: int = 300,
    learning_rate: float = 0.01,
    l2: float = 0.0001,
    threshold: float = 0.5,
    positive_weight: float = 1.0,
    use_reliability_features: bool = False,
) -> dict[str, Any]:
    if len(records) < 2:
        return {"available": False, "reason": "requires at least two records"}
    examples = []
    true_positive = false_positive = true_negative = false_negative = 0
    for index, record in enumerate(records):
        training = records[:index] + records[index + 1 :]
        reliability_context = (
            worker_reliability_context(training) if use_reliability_features else None
        )
        head, _ = train_latency_safe_head(
            training,
            router=router,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            threshold=threshold,
            positive_weight=positive_weight,
            use_reliability_features=use_reliability_features,
        )
        label = latency_safe_label(record)
        probability = head.probability(
            record,
            router=router,
            reliability_context=reliability_context,
        )
        predicted = float(probability >= threshold)
        true_positive += int(predicted == 1.0 and label == 1.0)
        false_positive += int(predicted == 1.0 and label == 0.0)
        true_negative += int(predicted == 0.0 and label == 0.0)
        false_negative += int(predicted == 0.0 and label == 1.0)
        examples.append(
            {
                "task_id": record["task_id"],
                "label": label,
                "probability": probability,
                "predicted": predicted,
            }
        )
    total = len(records)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "available": True,
        "task_count": total,
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "positive_count": true_positive + false_negative,
        "predicted_positive_count": true_positive + false_positive,
        "examples": examples,
    }
