from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .conditional_policy import worker_by_id
from .logits_router import LogitsRouter, scale_feature
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


def ranked_distribution(record: dict[str, Any], router: LogitsRouter) -> list[tuple[str, float]]:
    distribution = router.distribution(record)
    return sorted(distribution.items(), key=lambda item: item[1], reverse=True)


def collect_fallback_feature_names(records: list[dict[str, Any]], router: LogitsRouter) -> list[str]:
    names = {
        "bias",
        "top_probability",
        "second_probability",
        "first_second_margin",
        "router_entropy",
        "top_failed",
    }
    for record in records:
        names.update(extract_task_features(record))
        ranked = ranked_distribution(record, router)
        if ranked:
            names.add(f"top_worker_{feature_safe_name(ranked[0][0])}")
        if len(ranked) > 1:
            names.add(f"second_worker_{feature_safe_name(ranked[1][0])}")
    return sorted(names)


def fallback_features(record: dict[str, Any], router: LogitsRouter) -> dict[str, float]:
    ranked = ranked_distribution(record, router)
    probabilities = [probability for _, probability in ranked]
    top_worker_id = ranked[0][0] if ranked else ""
    second_worker_id = ranked[1][0] if len(ranked) > 1 else ""
    top_probability = ranked[0][1] if ranked else 0.0
    second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    top_worker = worker_by_id(record, top_worker_id) if top_worker_id else None

    features = {
        "bias": 1.0,
        "top_probability": top_probability,
        "second_probability": second_probability,
        "first_second_margin": top_probability - second_probability,
        "router_entropy": entropy(probabilities),
        "top_failed": float(top_worker is not None and not bool(top_worker["passed"])),
    }
    features.update(extract_task_features(record))
    if top_worker_id:
        features[f"top_worker_{feature_safe_name(top_worker_id)}"] = 1.0
    if second_worker_id:
        features[f"second_worker_{feature_safe_name(second_worker_id)}"] = 1.0
    return features


def fallback_feature_vector(
    record: dict[str, Any],
    router: LogitsRouter,
    feature_names: list[str],
) -> list[float]:
    features = fallback_features(record, router)
    return [scale_feature(name, features.get(name, 0.0)) for name in feature_names]


def fallback_label(
    record: dict[str, Any],
    router: LogitsRouter,
    label_mode: str = "rescue",
    teacher_margin: float = 0.1,
) -> float:
    ranked = ranked_distribution(record, router)
    if len(ranked) < 2:
        return 0.0
    top_worker = worker_by_id(record, ranked[0][0])
    second_worker = worker_by_id(record, ranked[1][0])
    if label_mode == "margin-gate":
        margin = float(ranked[0][1]) - float(ranked[1][1])
        return float((not bool(top_worker["passed"])) and margin <= teacher_margin)
    return float((not bool(top_worker["passed"])) and bool(second_worker["passed"]))


@dataclass
class FallbackHead:
    feature_names: list[str]
    weights: list[float]
    threshold: float = 0.5

    def logit(self, record: dict[str, Any], router: LogitsRouter) -> float:
        vector = fallback_feature_vector(record, router, self.feature_names)
        return sum(weight * value for weight, value in zip(self.weights, vector, strict=True))

    def probability(self, record: dict[str, Any], router: LogitsRouter) -> float:
        return sigmoid(self.logit(record, router))

    def should_fallback(self, record: dict[str, Any], router: LogitsRouter) -> bool:
        return self.probability(record, router) >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "fallback-logit-head",
            "feature_names": self.feature_names,
            "weights": self.weights,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FallbackHead":
        return cls(
            feature_names=list(payload["feature_names"]),
            weights=[float(value) for value in payload["weights"]],
            threshold=float(payload.get("threshold", 0.5)),
        )


def train_fallback_head(
    records: list[dict[str, Any]],
    router: LogitsRouter,
    epochs: int = 300,
    learning_rate: float = 0.01,
    l2: float = 0.0001,
    threshold: float = 0.5,
    label_mode: str = "rescue",
    teacher_margin: float = 0.1,
) -> tuple[FallbackHead, list[dict[str, float]]]:
    feature_names = collect_fallback_feature_names(records, router)
    head = FallbackHead(
        feature_names=feature_names,
        weights=[0.0 for _ in feature_names],
        threshold=threshold,
    )
    history = []
    if not records:
        return head, history

    for epoch in range(epochs):
        total_loss = 0.0
        for record in records:
            vector = fallback_feature_vector(record, router, feature_names)
            target = fallback_label(
                record,
                router,
                label_mode=label_mode,
                teacher_margin=teacher_margin,
            )
            predicted = sigmoid(
                sum(weight * value for weight, value in zip(head.weights, vector, strict=True))
            )
            total_loss += binary_cross_entropy(target, predicted)
            error = predicted - target
            for index, feature_value in enumerate(vector):
                penalty = l2 * head.weights[index]
                gradient = error * feature_value + penalty
                head.weights[index] -= learning_rate * gradient
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "mean_loss": total_loss / len(records),
                }
            )
    return head, history


def evaluate_fallback_head(
    records: list[dict[str, Any]],
    router: LogitsRouter,
    head: FallbackHead,
    max_attempts: int = 2,
) -> dict[str, Any]:
    if max_attempts != 2:
        raise ValueError("fallback logit head currently supports max_attempts=2")

    solved = 0
    matched_target = 0
    solvable_task_count = 0
    solvable_solved = 0
    solvable_matched_target = 0
    fallback_opportunities = 0
    fallbacks_taken = 0
    useful_fallbacks = 0
    total_latency = 0
    total_target_latency = 0
    total_latency_regret = 0
    examples = []

    for record in records:
        ranked = ranked_distribution(record, router)
        if not ranked:
            continue
        target_worker = worker_by_id(record, record["target_worker_id"])
        target_latency = int(target_worker["latency_ms"])
        top_worker = worker_by_id(record, ranked[0][0])
        final_worker = top_worker
        task_latency = int(top_worker["latency_ms"])
        attempts = [
            {
                "worker_id": top_worker["worker_id"],
                "passed": bool(top_worker["passed"]),
                "latency_ms": int(top_worker["latency_ms"]),
                "router_probability": ranked[0][1],
            }
        ]
        fallback_probability = head.probability(record, router)
        fallback_label_value = fallback_label(record, router)
        if not bool(top_worker["passed"]) and len(ranked) > 1:
            fallback_opportunities += 1
            if head.should_fallback(record, router):
                second_worker = worker_by_id(record, ranked[1][0])
                fallbacks_taken += 1
                useful_fallbacks += int(bool(second_worker["passed"]))
                final_worker = second_worker
                task_latency += int(second_worker["latency_ms"])
                attempts.append(
                    {
                        "worker_id": second_worker["worker_id"],
                        "passed": bool(second_worker["passed"]),
                        "latency_ms": int(second_worker["latency_ms"]),
                        "router_probability": ranked[1][1],
                    }
                )

        task_solved = bool(final_worker["passed"])
        final_worker_id = str(final_worker["worker_id"])
        solvable = any(bool(worker["passed"]) for worker in record["workers"])
        solved += int(task_solved)
        matched_target += int(final_worker_id == record["target_worker_id"])
        if solvable:
            solvable_task_count += 1
            solvable_solved += int(task_solved)
            solvable_matched_target += int(final_worker_id == record["target_worker_id"])
        total_latency += task_latency
        total_target_latency += target_latency
        total_latency_regret += max(0, task_latency - target_latency)
        examples.append(
            {
                "task_id": record["task_id"],
                "target_worker_id": record["target_worker_id"],
                "attempts": attempts,
                "fallback_probability": fallback_probability,
                "fallback_label": fallback_label_value,
                "final_worker_id": final_worker_id,
                "solved": task_solved,
                "matched_target": final_worker_id == record["target_worker_id"],
            }
        )

    task_count = len(records)
    return {
        "policy": "fallback-logit-head",
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
        "mean_latency_ms": total_latency / task_count if task_count else 0.0,
        "mean_target_latency_ms": total_target_latency / task_count if task_count else 0.0,
        "mean_latency_regret_ms": total_latency_regret / task_count if task_count else 0.0,
        "examples": examples,
    }
