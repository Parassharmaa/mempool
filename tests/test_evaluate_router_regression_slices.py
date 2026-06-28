import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_router_regression_slices import evaluate_slices


class EvaluateRouterRegressionSlicesTest(unittest.TestCase):
    def test_reports_failed_slice_when_solvable_pass_rate_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "slice.jsonl"
            model = root / "model.json"
            registry = root / "registry.json"
            manifest = root / "manifest.json"
            dataset.write_text(
                json.dumps(
                    {
                        "task_id": "t1",
                        "benchmark_id": "b",
                        "task_family": "code",
                        "prompt": "solve",
                        "prompt_features": {"length_chars": 5},
                        "target_worker_id": "solver",
                        "target_distribution": {"fast": 0.1, "solver": 0.9},
                        "workers": [
                            {
                                "worker_id": "fast",
                                "model": "fast",
                                "passed": False,
                                "score": 0.0,
                                "latency_ms": 1,
                                "cost_usd": 0.0,
                                "failure_mode": "test_failure",
                                "reward": -0.05,
                                "target_probability": 0.1,
                            },
                            {
                                "worker_id": "solver",
                                "model": "solver",
                                "passed": True,
                                "score": 1.0,
                                "latency_ms": 10,
                                "cost_usd": 0.0,
                                "failure_mode": None,
                                "reward": 0.95,
                                "target_probability": 0.9,
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            model.write_text(
                json.dumps(
                    {
                        "router": {
                            "worker_ids": ["fast", "solver"],
                            "feature_names": ["bias"],
                            "weights": [[1.0], [0.0]],
                        }
                    }
                ),
                encoding="utf-8",
            )
            registry.write_text(
                json.dumps({"active": {"model": str(model), "dataset": str(dataset)}}),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "slices": [
                            {
                                "id": "slice",
                                "dataset": str(dataset),
                                "expected_solvable_task_count": 1,
                                "minimum_solvable_pass_at_1": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_slices(registry, manifest)

        self.assertFalse(report["passed"])
        self.assertFalse(report["results"][0]["passed"])
        self.assertEqual(report["results"][0]["evaluation"]["solvable_pass_at_1"], 0.0)

    def test_conditional_policy_can_pass_slice_after_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "slice.jsonl"
            model = root / "model.json"
            registry = root / "registry.json"
            manifest = root / "manifest.json"
            record = {
                "task_id": "t1",
                "benchmark_id": "b",
                "task_family": "code",
                "prompt": "solve",
                "prompt_features": {"length_chars": 5},
                "target_worker_id": "solver",
                "target_distribution": {"fast": 0.1, "solver": 0.9},
                "workers": [
                    {
                        "worker_id": "fast",
                        "model": "fast",
                        "passed": False,
                        "score": 0.0,
                        "latency_ms": 1,
                        "cost_usd": 0.0,
                        "failure_mode": "test_failure",
                        "reward": -0.05,
                        "target_probability": 0.1,
                    },
                    {
                        "worker_id": "solver",
                        "model": "solver",
                        "passed": True,
                        "score": 1.0,
                        "latency_ms": 10,
                        "cost_usd": 0.0,
                        "failure_mode": None,
                        "reward": 0.95,
                        "target_probability": 0.9,
                    },
                ],
            }
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
            model.write_text(
                json.dumps(
                    {
                        "router": {
                            "worker_ids": ["fast", "solver"],
                            "feature_names": ["bias"],
                            "weights": [[1.0], [0.0]],
                        }
                    }
                ),
                encoding="utf-8",
            )
            registry.write_text(
                json.dumps({"active": {"model": str(model), "dataset": str(dataset)}}),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "slices": [
                            {
                                "id": "slice",
                                "dataset": str(dataset),
                                "expected_solvable_task_count": 1,
                                "minimum_solvable_pass_at_1": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_slices(registry, manifest, policy="conditional", max_attempts=2)

        self.assertTrue(report["passed"])
        self.assertEqual(report["results"][0]["evaluation"]["solvable_pass_at_1"], 1.0)

    def test_gated_policy_can_pass_slice_after_low_margin_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "slice.jsonl"
            model = root / "model.json"
            registry = root / "registry.json"
            manifest = root / "manifest.json"
            record = {
                "task_id": "t1",
                "benchmark_id": "b",
                "task_family": "code",
                "prompt": "solve",
                "prompt_features": {"length_chars": 5},
                "target_worker_id": "solver",
                "target_distribution": {"fast": 0.1, "solver": 0.9},
                "workers": [
                    {
                        "worker_id": "fast",
                        "model": "fast",
                        "passed": False,
                        "score": 0.0,
                        "latency_ms": 1,
                        "cost_usd": 0.0,
                        "failure_mode": "test_failure",
                        "reward": -0.05,
                        "target_probability": 0.1,
                    },
                    {
                        "worker_id": "solver",
                        "model": "solver",
                        "passed": True,
                        "score": 1.0,
                        "latency_ms": 10,
                        "cost_usd": 0.0,
                        "failure_mode": None,
                        "reward": 0.95,
                        "target_probability": 0.9,
                    },
                ],
            }
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
            model.write_text(
                json.dumps(
                    {
                        "router": {
                            "worker_ids": ["fast", "solver"],
                            "feature_names": ["bias"],
                            "weights": [[0.1], [0.0]],
                        }
                    }
                ),
                encoding="utf-8",
            )
            registry.write_text(
                json.dumps({"active": {"model": str(model), "dataset": str(dataset)}}),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "slices": [
                            {
                                "id": "slice",
                                "dataset": str(dataset),
                                "expected_solvable_task_count": 1,
                                "minimum_solvable_pass_at_1": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_slices(
                registry,
                manifest,
                policy="gated",
                max_attempts=2,
                max_first_second_margin=0.1,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["results"][0]["evaluation"]["fallbacks_taken"], 1)


if __name__ == "__main__":
    unittest.main()
