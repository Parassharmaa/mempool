from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    prompt: str
    family: str = "code"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    task_id: str
    passed: bool
    score: float
    latency_ms: int | None = None
    cost_usd: float | None = None
    failure_mode: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    id: str

    def load_tasks(self, limit: int | None = None) -> tuple[BenchmarkTask, ...]:
        """Load benchmark tasks in deterministic order."""

    def evaluate_output(self, task: BenchmarkTask, output: str) -> BenchmarkResult:
        """Evaluate one output for one task."""


@dataclass(frozen=True)
class BenchmarkPlan:
    benchmark_id: str
    status: str
    reason_for_choice: str
    ladder: tuple[dict[str, str | int], ...]
    baselines: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    run_rules: tuple[str, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            benchmark_id=data["benchmark_id"],
            status=data["status"],
            reason_for_choice=data["reason_for_choice"],
            ladder=tuple(data["ladder"]),
            baselines=tuple(data["baselines"]),
            primary_metrics=tuple(data["primary_metrics"]),
            secondary_metrics=tuple(data["secondary_metrics"]),
            run_rules=tuple(data["run_rules"]),
        )

    def stage(self, name: str) -> dict[str, str | int]:
        for stage in self.ladder:
            if stage.get("name") == name:
                return stage
        raise KeyError(f"unknown benchmark stage: {name}")
