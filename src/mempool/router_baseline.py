from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .task_features import extract_task_features, feature_distance


@dataclass(frozen=True)
class RouterEvaluation:
    policy: str
    task_count: int
    matched_target: int
    solved: int
    total_latency_ms: int
    total_cost_usd: float
    total_target_latency_ms: int = 0
    total_latency_regret_ms: int = 0
    solvable_task_count: int = 0
    solvable_solved: int = 0
    solvable_matched_target: int = 0

    @property
    def target_accuracy(self) -> float:
        return self.matched_target / self.task_count if self.task_count else 0.0

    @property
    def pass_at_1(self) -> float:
        return self.solved / self.task_count if self.task_count else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.task_count if self.task_count else 0.0

    @property
    def mean_target_latency_ms(self) -> float:
        return self.total_target_latency_ms / self.task_count if self.task_count else 0.0

    @property
    def mean_latency_regret_ms(self) -> float:
        return self.total_latency_regret_ms / self.task_count if self.task_count else 0.0

    @property
    def solvable_pass_at_1(self) -> float:
        return self.solvable_solved / self.solvable_task_count if self.solvable_task_count else 0.0

    @property
    def solvable_target_accuracy(self) -> float:
        return self.solvable_matched_target / self.solvable_task_count if self.solvable_task_count else 0.0

    @property
    def cost_per_solved_task(self) -> float | None:
        if not self.solved:
            return None
        return self.total_cost_usd / self.solved


class FamilyRouter:
    def __init__(self, default_worker_id: str, family_to_worker: dict[str, str]) -> None:
        self.default_worker_id = default_worker_id
        self.family_to_worker = family_to_worker

    @classmethod
    def train(cls, records: list[dict[str, Any]]) -> "FamilyRouter":
        by_family: dict[str, Counter[str]] = defaultdict(Counter)
        global_counts: Counter[str] = Counter()
        for record in records:
            family = record["task_family"]
            worker_id = record["target_worker_id"]
            by_family[family][worker_id] += 1
            global_counts[worker_id] += 1
        default_worker_id = global_counts.most_common(1)[0][0]
        family_to_worker = {
            family: counts.most_common(1)[0][0] for family, counts in by_family.items()
        }
        return cls(default_worker_id=default_worker_id, family_to_worker=family_to_worker)

    def predict(self, record: dict[str, Any]) -> str:
        return self.family_to_worker.get(record["task_family"], self.default_worker_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "family-router",
            "default_worker_id": self.default_worker_id,
            "family_to_worker": self.family_to_worker,
        }


class NearestNeighborRouter:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.features = [extract_task_features(record) for record in records]
        self.default_worker_id = records[0]["target_worker_id"] if records else ""

    @classmethod
    def train(cls, records: list[dict[str, Any]]) -> "NearestNeighborRouter":
        return cls(records)

    def predict(self, record: dict[str, Any]) -> str:
        if not self.records:
            return self.default_worker_id
        features = extract_task_features(record)
        best_index = min(
            range(len(self.records)),
            key=lambda index: (feature_distance(features, self.features[index]), self.records[index].get("task_id", "")),
        )
        return str(self.records[best_index]["target_worker_id"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "nearest-neighbor-router",
            "task_count": len(self.records),
            "default_worker_id": self.default_worker_id,
        }


def worker_by_id(record: dict[str, Any], worker_id: str) -> dict[str, Any]:
    for worker in record["workers"]:
        if worker["worker_id"] == worker_id:
            return worker
    raise KeyError(worker_id)


def evaluate_policy(
    records: list[dict[str, Any]],
    policy: str,
    predictions: list[str],
) -> RouterEvaluation:
    matched_target = 0
    solved = 0
    total_latency = 0
    total_target_latency = 0
    total_latency_regret = 0
    total_cost = 0.0
    solvable_task_count = 0
    solvable_solved = 0
    solvable_matched_target = 0
    for record, prediction in zip(records, predictions, strict=True):
        worker = worker_by_id(record, prediction)
        target_worker = worker_by_id(record, record["target_worker_id"])
        solvable = any(bool(item["passed"]) for item in record["workers"])
        matched_target += int(prediction == record["target_worker_id"])
        passed = bool(worker["passed"])
        solved += int(passed)
        latency = int(worker["latency_ms"])
        target_latency = int(target_worker["latency_ms"])
        total_latency += latency
        total_target_latency += target_latency
        total_latency_regret += max(0, latency - target_latency)
        total_cost += float(worker["cost_usd"] or 0.0)
        if solvable:
            solvable_task_count += 1
            solvable_solved += int(passed)
            solvable_matched_target += int(prediction == record["target_worker_id"])
    return RouterEvaluation(
        policy=policy,
        task_count=len(records),
        matched_target=matched_target,
        solved=solved,
        total_latency_ms=total_latency,
        total_cost_usd=total_cost,
        total_target_latency_ms=total_target_latency,
        total_latency_regret_ms=total_latency_regret,
        solvable_task_count=solvable_task_count,
        solvable_solved=solvable_solved,
        solvable_matched_target=solvable_matched_target,
    )


def strongest_worker(records: list[dict[str, Any]]) -> str:
    scores: Counter[str] = Counter()
    latencies: dict[str, list[int]] = defaultdict(list)
    for record in records:
        for worker in record["workers"]:
            scores[worker["worker_id"]] += int(worker["passed"])
            latencies[worker["worker_id"]].append(int(worker["latency_ms"]))
    return min(
        scores,
        key=lambda worker_id: (-scores[worker_id], sum(latencies[worker_id]) / len(latencies[worker_id]), worker_id),
    )


def fastest_worker(records: list[dict[str, Any]]) -> str:
    latencies: dict[str, list[int]] = defaultdict(list)
    for record in records:
        for worker in record["workers"]:
            latencies[worker["worker_id"]].append(int(worker["latency_ms"]))
    return min(latencies, key=lambda worker_id: sum(latencies[worker_id]) / len(latencies[worker_id]))


def leave_one_out_predictions(records: list[dict[str, Any]], router_type: str = "family") -> list[str]:
    predictions = []
    for index, record in enumerate(records):
        training = records[:index] + records[index + 1 :]
        if router_type == "nearest-neighbor":
            router = NearestNeighborRouter.train(training)
        else:
            router = FamilyRouter.train(training)
        predictions.append(router.predict(record))
    return predictions
