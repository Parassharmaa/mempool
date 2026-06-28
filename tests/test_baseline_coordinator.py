import unittest

from mempool.coordinator import BaselineCoordinator
from mempool.orchestration import WorkflowKind
from mempool.task import Task, TaskConstraints
from mempool.worker import Worker, WorkerPool


class BaselineCoordinatorTest(unittest.TestCase):
    def test_routes_to_best_matching_available_worker(self) -> None:
        task = Task(id="t1", prompt="Solve this proof.", family="math")
        pool = WorkerPool(
            workers=(
                Worker(id="general", adapter="local", strengths=("general",)),
                Worker(id="math", adapter="hosted", strengths=("math",)),
            )
        )

        plan = BaselineCoordinator().plan(task, pool)

        self.assertEqual(plan.kind, WorkflowKind.ROUTE)
        self.assertEqual(plan.worker_ids, ("math",))

    def test_uses_verifier_when_required(self) -> None:
        task = Task(
            id="t2",
            prompt="Review this answer.",
            family="research",
            constraints=TaskConstraints(require_verification=True),
        )
        pool = WorkerPool(
            workers=(
                Worker(id="researcher", adapter="hosted", strengths=("research",)),
                Worker(id="verifier", adapter="local", strengths=("verification",)),
            )
        )

        plan = BaselineCoordinator().plan(task, pool)

        self.assertEqual(plan.kind, WorkflowKind.REPAIR)
        self.assertEqual(plan.worker_ids, ("researcher",))
        self.assertEqual(plan.verifier_id, "verifier")

    def test_abstains_when_constraints_block_all_workers(self) -> None:
        task = Task(
            id="t3",
            prompt="Answer privately.",
            constraints=TaskConstraints(blocked_worker_ids=("blocked",)),
        )
        pool = WorkerPool(
            workers=(Worker(id="blocked", adapter="hosted", strengths=("general",)),)
        )

        plan = BaselineCoordinator().plan(task, pool)

        self.assertEqual(plan.kind, WorkflowKind.ABSTAIN)


if __name__ == "__main__":
    unittest.main()
