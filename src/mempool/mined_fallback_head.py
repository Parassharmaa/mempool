from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .logits_router import scale_feature
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


def mined_label(record: dict[str, Any], label_field: str = "useful_any_fallback") -> float:
    return float(bool(record.get(label_field, False)))


def mined_fallback_features(record: dict[str, Any]) -> dict[str, float]:
    features = {
        "bias": 1.0,
        "top_probability": float(record.get("top_probability", 0.0)),
        "second_probability": float(record.get("second_probability", 0.0)),
        "first_second_margin": float(record.get("first_second_margin", 0.0)),
        "top_latency_s": float(record.get("top_latency_ms", 0.0)) / 1000.0,
        "second_latency_s": float(record.get("second_latency_ms", 0.0)) / 1000.0,
        "alternate_count": float(record.get("alternate_count", 0.0)),
    }
    features.update(extract_task_features(record))
    top_worker_id = str(record.get("top_worker_id", ""))
    second_worker_id = str(record.get("second_worker_id", ""))
    if top_worker_id:
        features[f"top_worker_{feature_safe_name(top_worker_id)}"] = 1.0
    if second_worker_id:
        features[f"second_worker_{feature_safe_name(second_worker_id)}"] = 1.0
    return features


def collect_feature_names(records: list[dict[str, Any]]) -> list[str]:
    names = set()
    for record in records:
        names.update(mined_fallback_features(record))
    return sorted(names)


def feature_vector(record: dict[str, Any], feature_names: list[str]) -> list[float]:
    features = mined_fallback_features(record)
    return [scale_feature(name, features.get(name, 0.0)) for name in feature_names]


@dataclass
class MinedFallbackHead:
    feature_names: list[str]
    weights: list[float]
    threshold: float = 0.5
    label_field: str = "useful_any_fallback"

    def logit(self, record: dict[str, Any]) -> float:
        vector = feature_vector(record, self.feature_names)
        return sum(weight * value for weight, value in zip(self.weights, vector, strict=True))

    def probability(self, record: dict[str, Any]) -> float:
        return sigmoid(self.logit(record))

    def should_fallback(self, record: dict[str, Any]) -> bool:
        return self.probability(record) >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "mined-fallback-logit-head",
            "feature_names": self.feature_names,
            "weights": self.weights,
            "threshold": self.threshold,
            "label_field": self.label_field,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MinedFallbackHead":
        return cls(
            feature_names=list(payload["feature_names"]),
            weights=[float(value) for value in payload["weights"]],
            threshold=float(payload.get("threshold", 0.5)),
            label_field=str(payload.get("label_field", "useful_any_fallback")),
        )


def train_mined_fallback_head(
    records: list[dict[str, Any]],
    epochs: int = 500,
    learning_rate: float = 0.02,
    l2: float = 0.0001,
    threshold: float = 0.5,
    label_field: str = "useful_any_fallback",
    positive_weight: float | None = None,
) -> tuple[MinedFallbackHead, list[dict[str, float]]]:
    feature_names = collect_feature_names(records)
    head = MinedFallbackHead(
        feature_names=feature_names,
        weights=[0.0 for _ in feature_names],
        threshold=threshold,
        label_field=label_field,
    )
    history = []
    if not records:
        return head, history

    positive_count = sum(1 for record in records if mined_label(record, label_field))
    negative_count = len(records) - positive_count
    if positive_weight is None:
        positive_weight = negative_count / positive_count if positive_count else 1.0

    for epoch in range(epochs):
        total_loss = 0.0
        for record in records:
            vector = feature_vector(record, feature_names)
            target = mined_label(record, label_field)
            predicted = sigmoid(
                sum(weight * value for weight, value in zip(head.weights, vector, strict=True))
            )
            sample_weight = positive_weight if target else 1.0
            total_loss += sample_weight * binary_cross_entropy(target, predicted)
            error = sample_weight * (predicted - target)
            for index, feature_value in enumerate(vector):
                gradient = error * feature_value + l2 * head.weights[index]
                head.weights[index] -= learning_rate * gradient
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "mean_loss": total_loss / len(records),
                }
            )
    return head, history


def classification_metrics(
    records: list[dict[str, Any]],
    probabilities: list[float],
    threshold: float,
    label_field: str = "useful_any_fallback",
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    examples = []
    for record, probability in zip(records, probabilities, strict=True):
        predicted = probability >= threshold
        actual = bool(mined_label(record, label_field))
        tp += int(predicted and actual)
        fp += int(predicted and not actual)
        tn += int((not predicted) and (not actual))
        fn += int((not predicted) and actual)
        examples.append(
            {
                "task_id": record["task_id"],
                "probability": probability,
                "predicted_fallback": predicted,
                "actual_useful_fallback": actual,
                "top_worker_id": record.get("top_worker_id"),
                "best_ranked_alternate_worker_id": record.get("best_ranked_alternate_worker_id"),
            }
        )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    task_count = len(records)
    return {
        "task_count": task_count,
        "positive_count": tp + fn,
        "negative_count": tn + fp,
        "predicted_positive_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / task_count if task_count else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "examples": examples,
    }


def evaluate_head(
    records: list[dict[str, Any]],
    head: MinedFallbackHead,
) -> dict[str, Any]:
    probabilities = [head.probability(record) for record in records]
    return classification_metrics(
        records,
        probabilities,
        threshold=head.threshold,
        label_field=head.label_field,
    )


def select_threshold(
    records: list[dict[str, Any]],
    probabilities: list[float],
    thresholds: list[float],
    label_field: str = "useful_any_fallback",
) -> tuple[float, dict[str, Any]]:
    if not thresholds:
        raise ValueError("at least one threshold is required")
    candidates = [
        (threshold, classification_metrics(records, probabilities, threshold, label_field))
        for threshold in thresholds
    ]
    return max(
        candidates,
        key=lambda item: (
            float(item[1]["f1"]),
            float(item[1]["recall"]),
            float(item[1]["precision"]),
            float(item[1]["accuracy"]),
            -float(item[1]["predicted_positive_count"]),
        ),
    )


def leave_one_out_evaluation(
    records: list[dict[str, Any]],
    thresholds: list[float],
    epochs: int = 500,
    learning_rate: float = 0.02,
    l2: float = 0.0001,
    label_field: str = "useful_any_fallback",
) -> dict[str, Any]:
    if len(records) < 2:
        return {"task_count": len(records), "folds": [], "metrics": {}}

    probabilities = []
    folds = []
    ordered_records = []
    for index, heldout in enumerate(records):
        train_records = records[:index] + records[index + 1 :]
        trained, _ = train_mined_fallback_head(
            train_records,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            label_field=label_field,
        )
        train_probabilities = [trained.probability(record) for record in train_records]
        selected_threshold, train_metrics = select_threshold(
            train_records,
            train_probabilities,
            thresholds,
            label_field=label_field,
        )
        heldout_probability = trained.probability(heldout)
        probabilities.append(heldout_probability)
        ordered_records.append(heldout)
        folds.append(
            {
                "task_id": heldout["task_id"],
                "selected_threshold": selected_threshold,
                "heldout_probability": heldout_probability,
                "actual_useful_fallback": bool(mined_label(heldout, label_field)),
                "train_metrics": {
                    key: value
                    for key, value in train_metrics.items()
                    if key != "examples"
                },
            }
        )

    selected_threshold, metrics = select_threshold(
        ordered_records,
        probabilities,
        thresholds,
        label_field=label_field,
    )
    return {
        "task_count": len(records),
        "selected_global_threshold": selected_threshold,
        "metrics": metrics,
        "folds": folds,
    }
