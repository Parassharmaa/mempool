import json
import tempfile
import unittest
from pathlib import Path

from tools.select_gated_fallback_threshold import policy_payload, select_threshold


def routing_record(task_id: str, prompt: str = "solve") -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "b",
        "task_family": "code",
        "prompt": prompt,
        "prompt_features": {"length_chars": len(prompt)},
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


class SelectGatedFallbackThresholdTest(unittest.TestCase):
    def test_selects_smallest_margin_that_preserves_regression_and_active_solve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "active.jsonl"
            slice_dataset = root / "slice.jsonl"
            model = root / "model.json"
            registry = root / "registry.json"
            manifest = root / "manifest.json"

            dataset.write_text(json.dumps(routing_record("active")) + "\n", encoding="utf-8")
            slice_dataset.write_text(json.dumps(routing_record("slice")) + "\n", encoding="utf-8")
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
                                "dataset": str(slice_dataset),
                                "expected_solvable_task_count": 1,
                                "minimum_solvable_pass_at_1": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            selection = select_threshold(
                registry,
                dataset,
                manifest,
                margins=[0.01, 0.05, 0.1, 0.2],
            )
            policy = policy_payload(selection)

        self.assertTrue(selection["valid"])
        self.assertEqual(selection["selected"]["max_first_second_margin"], 0.05)
        self.assertFalse(selection["candidates"][0]["eligible"])
        self.assertEqual(selection["selected"]["active_evaluation"]["solvable_pass_at_1"], 1.0)
        self.assertEqual(policy["policy"], "gated-fallback")
        self.assertEqual(policy["max_first_second_margin"], 0.05)


if __name__ == "__main__":
    unittest.main()
