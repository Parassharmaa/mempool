from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskConstraints:
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None
    allowed_worker_ids: tuple[str, ...] = ()
    blocked_worker_ids: tuple[str, ...] = ()
    require_provenance: bool = True
    require_verification: bool = False
    data_policy: str = "default"

    def allows_worker(self, worker_id: str) -> bool:
        if self.allowed_worker_ids and worker_id not in self.allowed_worker_ids:
            return False
        return worker_id not in self.blocked_worker_ids


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    family: str = "general"
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    eval_hints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
