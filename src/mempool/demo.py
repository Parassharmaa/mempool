from __future__ import annotations

from dataclasses import asdict

from .coordinator import BaselineCoordinator
from .ledger import JsonlLedger, LedgerEvent
from .task import Task, TaskConstraints
from .worker import Worker, WorkerPool


def main() -> None:
    task = Task(
        id="demo-001",
        prompt="Design a minimal eval for a learned orchestration policy.",
        family="research",
        constraints=TaskConstraints(require_verification=True),
    )
    pool = WorkerPool(
        workers=(
            Worker(
                id="frontier-researcher",
                adapter="hosted-chat",
                strengths=("research", "general"),
                expected_latency_ms=8000,
                expected_cost_usd=0.05,
            ),
            Worker(
                id="local-verifier",
                adapter="local",
                strengths=("verification", "general"),
                expected_latency_ms=2000,
                expected_cost_usd=0.0,
            ),
        )
    )
    plan = BaselineCoordinator().plan(task, pool)
    ledger = JsonlLedger("research/logs/demo.jsonl")
    ledger.append(
        LedgerEvent(
            type="workflow_planned",
            task_id=task.id,
            payload={"task": asdict(task), "plan": asdict(plan)},
        )
    )
    print(asdict(plan))


if __name__ == "__main__":
    main()
