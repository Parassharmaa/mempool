import unittest

from tools.merge_routing_datasets import assert_merge_ready


class MergeRoutingDatasetsTest(unittest.TestCase):
    def test_assert_merge_ready_accepts_stable_dataset(self) -> None:
        report = assert_merge_ready(
            [
                {
                    "task_id": "task",
                    "benchmark_id": "bench",
                    "task_family": "code",
                    "prompt": "p",
                    "prompt_features": {},
                    "target_worker_id": "strong",
                    "target_distribution": {"strong": 1.0},
                    "workers": [
                        {
                            "worker_id": "strong",
                            "model": "m",
                            "passed": True,
                            "score": 1.0,
                            "latency_ms": 1,
                            "cost_usd": 0.0,
                            "failure_mode": None,
                            "reward": 1.0,
                            "target_probability": 1.0,
                            "pass_rate": 1.0,
                        }
                    ],
                }
            ]
        )

        self.assertTrue(report["ready_to_merge"])

    def test_assert_merge_ready_rejects_unstable_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "not merge-ready"):
            assert_merge_ready(
                [
                    {
                        "task_id": "task",
                        "benchmark_id": "bench",
                        "task_family": "code",
                        "prompt": "p",
                        "prompt_features": {},
                        "target_worker_id": "maybe",
                        "target_distribution": {"maybe": 1.0},
                        "workers": [
                            {
                                "worker_id": "maybe",
                                "model": "m",
                                "passed": True,
                                "score": 0.5,
                                "latency_ms": 1,
                                "cost_usd": 0.0,
                                "failure_mode": "mixed_samples",
                                "reward": 0.5,
                                "target_probability": 1.0,
                                "pass_rate": 0.5,
                            }
                        ],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
