from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class WorkflowKind(StrEnum):
    DIRECT = "direct"
    ROUTE = "route"
    COMPARE = "compare"
    DECOMPOSE = "decompose"
    REPAIR = "repair"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class WorkflowPlan:
    kind: WorkflowKind
    worker_ids: tuple[str, ...] = ()
    verifier_id: str | None = None
    rationale: str = ""
    expected_cost_usd: float | None = None
    expected_latency_ms: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)
