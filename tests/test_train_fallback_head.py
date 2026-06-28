import json
import tempfile
import unittest
from pathlib import Path

from tools.train_fallback_head import select_threshold, selected_head_payload


def routing_record(task_id: str, second_passes: bool = True) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "b",
        "task_family": "bigcodebench_hard",
        "prompt": f"solve {task_id}",
        "prompt_features": {"categories": [], "libraries": [], "missing_libraries": []},
        "target_worker_id": "solver" if second_passes else "fast",
        "target_distribution": {"fast": 0.2, "solver": 0.8},
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
                "target_probability": 0.2,
            },
            {
                "worker_id": "solver",
                "model": "solver",
                "passed": second_passes,
                "score": 1.0 if second_passes else 0.0,
                "latency_ms": 10,
                "cost_usd": 0.0,
                "failure_mode": None if second_passes else "test_failure",
                "reward": 0.95 if second_passes else -0.05,
                "target_probability": 0.8,
            },
        ],
    }


class TrainFallbackHeadToolTest(unittest.TestCase):
    def test_select_threshold_writes_model_ready_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "active.jsonl"
            slice_dataset = root / "slice.jsonl"
            model = root / "model.json"
            registry = root / "registry.json"
            manifest = root / "manifest.json"

            dataset.write_text(
                json.dumps(routing_record("active", True)) + "\n",
                encoding="utf-8",
            )
            slice_dataset.write_text(
                json.dumps(routing_record("slice", True)) + "\n",
                encoding="utf-8",
            )
            model.write_text(
                json.dumps(
                    {
                        "router": {
                            "worker_ids": ["fast", "solver"],
                            "feature_names": ["bias"],
                            "weights": [[1.0], [0.9]],
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
                thresholds=[0.0, 0.5],
                epochs=20,
                learning_rate=0.05,
                l2=0.0,
            )
            model_payload = selected_head_payload(selection)

        self.assertTrue(selection["valid"])
        self.assertEqual(selection["selected"]["threshold"], 0.0)
        self.assertEqual(model_payload["policy"], "fallback-logit-head")
        self.assertEqual(model_payload["head"]["threshold"], 0.0)
        self.assertTrue(model_payload["regression_passed"])


if __name__ == "__main__":
    unittest.main()
