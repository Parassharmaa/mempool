import unittest

from tools.build_routing_dataset import build_records


class BuildRoutingDatasetTest(unittest.TestCase):
    def test_groups_rows_and_builds_soft_targets(self) -> None:
        rows = [
            {
                "benchmark_id": "b",
                "task_id": "t1",
                "task_family": "code",
                "prompt": "write code",
                "worker_id": "slow-good",
                "model": "m1",
                "passed": True,
                "score": 1.0,
                "latency_ms": 100,
                "cost_usd": 0.0,
                "failure_mode": None,
            },
            {
                "benchmark_id": "b",
                "task_id": "t1",
                "task_family": "code",
                "prompt": "write code",
                "worker_id": "fast-bad",
                "model": "m2",
                "passed": False,
                "score": 0.0,
                "latency_ms": 10,
                "cost_usd": 0.0,
                "failure_mode": "test_failure",
            },
        ]

        records = build_records(rows)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_worker_id"], "slow-good")
        self.assertGreater(
            records[0]["target_distribution"]["slow-good"],
            records[0]["target_distribution"]["fast-bad"],
        )
        self.assertIn("categories", records[0]["prompt_features"])
        self.assertIn("libraries", records[0]["prompt_features"])


if __name__ == "__main__":
    unittest.main()
