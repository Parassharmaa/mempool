from __future__ import annotations

from .orchestration import WorkflowKind, WorkflowPlan
from .task import Task
from .worker import WorkerPool


class BaselineCoordinator:
    """Small rule baseline used before learned policies exist."""

    def plan(self, task: Task, pool: WorkerPool) -> WorkflowPlan:
        worker = pool.best_for(task)
        if worker is None:
            return WorkflowPlan(
                kind=WorkflowKind.ABSTAIN,
                rationale="No available worker satisfies the task constraints.",
            )

        if task.constraints.require_verification:
            verifier = self._pick_verifier(task, pool, exclude={worker.id})
            return WorkflowPlan(
                kind=WorkflowKind.REPAIR if verifier else WorkflowKind.ROUTE,
                worker_ids=(worker.id,),
                verifier_id=verifier,
                rationale="Selected best matching worker with verification requested.",
            )

        return WorkflowPlan(
            kind=WorkflowKind.ROUTE,
            worker_ids=(worker.id,),
            rationale="Selected highest scoring available worker.",
        )

    def _pick_verifier(
        self, task: Task, pool: WorkerPool, exclude: set[str]
    ) -> str | None:
        candidates = [
            worker
            for worker in pool.available_for(task)
            if worker.id not in exclude and "verification" in worker.strengths
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda worker: worker.score_for(task)).id
