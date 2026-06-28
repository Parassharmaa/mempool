import unittest

from mempool.routing_dataset import validate_routing_records


class RoutingDatasetTest(unittest.TestCase):
    def test_validates_probability_sum(self) -> None:
        records = [
            {
                "task_id": "t",
                "benchmark_id": "b",
                "task_family": "code",
                "prompt": "p",
                "prompt_features": {},
                "target_worker_id": "a",
                "target_distribution": {"a": 1.0},
                "workers": [
                    {
                        "worker_id": "a",
                        "model": "m",
                        "passed": True,
                        "score": 1.0,
                        "latency_ms": 1,
                        "cost_usd": 0.0,
                        "failure_mode": None,
                        "reward": 1.0,
                        "target_probability": 1.0,
                    }
                ],
            }
        ]

        self.assertEqual(validate_routing_records(records), [])


if __name__ == "__main__":
    unittest.main()
