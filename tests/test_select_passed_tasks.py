import unittest

from tools.select_passed_tasks_from_outcomes import passed_task_ids, select_tasks


class SelectPassedTasksTest(unittest.TestCase):
    def test_selects_passed_tasks_for_worker_in_source_order(self) -> None:
        tasks = [
            {"id": "task-a"},
            {"id": "task-b"},
            {"id": "task-c"},
        ]
        rows = [
            {"task_id": "task-c", "worker_id": "w1", "passed": True},
            {"task_id": "task-a", "worker_id": "w2", "passed": True},
            {"task_id": "task-b", "worker_id": "w1", "passed": False},
            {"task_id": "task-c", "worker_id": "w1", "passed": True},
        ]

        selected = select_tasks(tasks, rows, worker_id="w1")

        self.assertEqual(selected, [{"id": "task-c"}])

    def test_passed_task_ids_preserves_first_pass_order(self) -> None:
        rows = [
            {"task_id": "task-b", "worker_id": "w1", "passed": True},
            {"task_id": "task-a", "worker_id": "w1", "passed": True},
            {"task_id": "task-b", "worker_id": "w1", "passed": True},
        ]

        self.assertEqual(passed_task_ids(rows), ["task-b", "task-a"])


if __name__ == "__main__":
    unittest.main()
