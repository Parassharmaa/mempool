import unittest

from tools.select_worker_rejection_tasks import rejected_task_ids, select_tasks


class SelectWorkerRejectionTasksTest(unittest.TestCase):
    def test_selects_failed_worker_tasks_in_source_order(self) -> None:
        tasks = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        rows = [
            {"task_id": "c", "worker_id": "qwen", "passed": False, "latency_ms": 1000},
            {"task_id": "a", "worker_id": "other", "passed": False, "latency_ms": 1000},
            {"task_id": "b", "worker_id": "qwen", "passed": True, "latency_ms": 1000},
        ]

        self.assertEqual(rejected_task_ids(rows, worker_id="qwen"), ["c"])
        self.assertEqual(select_tasks(tasks, rows, worker_id="qwen"), [{"id": "c"}])

    def test_selects_slow_passes_over_threshold(self) -> None:
        rows = [
            {"task_id": "fast", "worker_id": "qwen", "passed": True, "latency_ms": 3000},
            {"task_id": "slow", "worker_id": "qwen", "passed": True, "latency_ms": 9000},
            {"task_id": "failed", "worker_id": "qwen", "passed": False, "latency_ms": 1000},
        ]

        self.assertEqual(
            rejected_task_ids(rows, worker_id="qwen", max_pass_latency_ms=6000),
            ["slow", "failed"],
        )

    def test_uses_mean_latency_when_needed(self) -> None:
        rows = [
            {"task_id": "slow", "worker_id": "qwen", "passed": True, "mean_latency_ms": 9000},
        ]

        self.assertEqual(
            rejected_task_ids(rows, worker_id="qwen", max_pass_latency_ms=6000),
            ["slow"],
        )


if __name__ == "__main__":
    unittest.main()
