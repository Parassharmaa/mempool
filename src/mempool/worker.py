from __future__ import annotations

from dataclasses import dataclass, field

from .task import Task


@dataclass(frozen=True)
class Worker:
    id: str
    adapter: str
    strengths: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ("text",)
    context_tokens: int | None = None
    expected_latency_ms: int | None = None
    expected_cost_usd: float | None = None
    data_policy: str = "default"
    tool_access: tuple[str, ...] = ()
    available: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    def score_for(self, task: Task) -> float:
        if not self.available or not task.constraints.allows_worker(self.id):
            return float("-inf")

        score = 0.0
        if task.family in self.strengths:
            score += 2.0
        if "general" in self.strengths:
            score += 0.5
        if task.constraints.max_latency_ms and self.expected_latency_ms:
            if self.expected_latency_ms > task.constraints.max_latency_ms:
                score -= 2.0
        if task.constraints.max_cost_usd and self.expected_cost_usd:
            if self.expected_cost_usd > task.constraints.max_cost_usd:
                score -= 2.0
        return score


@dataclass(frozen=True)
class WorkerPool:
    workers: tuple[Worker, ...]

    def available_for(self, task: Task) -> tuple[Worker, ...]:
        return tuple(
            worker
            for worker in self.workers
            if worker.available and task.constraints.allows_worker(worker.id)
        )

    def best_for(self, task: Task) -> Worker | None:
        candidates = self.available_for(task)
        if not candidates:
            return None
        return max(candidates, key=lambda worker: worker.score_for(task))
