import unittest

from tools.filter_outcome_jsonl_by_tasks import filter_rows


class FilterOutcomeJsonlByTasksTest(unittest.TestCase):
    def test_filter_rows_keeps_only_selected_tasks(self) -> None:
        rows = [
            {"task_id": "a", "worker_id": "w1"},
            {"task_id": "b", "worker_id": "w1"},
            {"task_id": "a", "worker_id": "w2"},
        ]

        filtered = filter_rows(rows, {"a"})

        self.assertEqual(filtered, [rows[0], rows[2]])


if __name__ == "__main__":
    unittest.main()
