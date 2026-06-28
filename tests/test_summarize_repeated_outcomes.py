import unittest

from tools.summarize_repeated_outcomes import summarize


class SummarizeRepeatedOutcomesTest(unittest.TestCase):
    def test_summarizes_pass_rate_by_task_worker_and_worker(self) -> None:
        rows = [
            {
                "task_id": "t1",
                "worker_id": "w1",
                "model": "m1",
                "sample_index": 1,
                "passed": False,
                "failure_mode": "test_failure",
                "latency_ms": 30,
            },
            {
                "task_id": "t1",
                "worker_id": "w1",
                "model": "m1",
                "sample_index": 0,
                "passed": True,
                "failure_mode": None,
                "latency_ms": 10,
            },
            {
                "task_id": "t2",
                "worker_id": "w1",
                "model": "m1",
                "sample_index": 0,
                "passed": True,
                "failure_mode": None,
                "latency_ms": 20,
            },
        ]

        report = summarize(rows)

        self.assertEqual(report["outcome_count"], 3)
        self.assertEqual(report["records"][0]["sample_passes"], [True, False])
        self.assertEqual(report["records"][0]["pass_rate"], 0.5)
        self.assertEqual(report["by_worker"][0]["pass_rate"], 2 / 3)
        self.assertEqual(report["candidate_task_ids"], ["t1", "t2"])
        self.assertEqual(report["universal_failure_task_ids"], [])
        self.assertEqual(report["by_task"][0]["best_worker_id"], "w1")

    def test_marks_universal_failure_tasks_as_not_convertible(self) -> None:
        rows = [
            {
                "task_id": "t1",
                "worker_id": "w1",
                "model": "m1",
                "sample_index": 0,
                "passed": False,
                "failure_mode": "test_failure",
                "latency_ms": 10,
            },
            {
                "task_id": "t1",
                "worker_id": "w2",
                "model": "m2",
                "sample_index": 0,
                "passed": False,
                "failure_mode": "test_failure",
                "latency_ms": 20,
            },
        ]

        report = summarize(rows)

        self.assertEqual(report["candidate_task_ids"], [])
        self.assertEqual(report["universal_failure_task_ids"], ["t1"])
        self.assertFalse(report["by_task"][0]["candidate_for_conversion"])
        self.assertIsNone(report["by_task"][0]["best_worker_id"])


if __name__ == "__main__":
    unittest.main()
