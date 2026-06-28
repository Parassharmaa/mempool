import json
import tempfile
import unittest
from pathlib import Path

from mempool.multi_head_orchestrator import train_multi_head_orchestrator
from mempool.orchestrator_runtime import (
    build_prompt_record,
    load_multi_head_orchestrator,
    normalize_prediction_record,
    predict_orchestration,
)


def example(task_id: str, keyword: str, target_worker: str) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "bigcodebench_hard",
        "prompt": f"Task mentions {keyword}",
        "dense_features": {"bias": 1.0, keyword: 1.0},
        "target": {
            "worker_distribution": {
                "qwen": 0.95 if target_worker == "qwen" else 0.05,
                "kimi": 0.95 if target_worker == "kimi" else 0.05,
            },
            "target_worker_id": target_worker,
            "workflow_distribution": {"direct": 1.0, "verify_then_fallback": 0.0},
            "workflow_kind": "direct",
            "verifier_probability": 0.1,
            "abstain_probability": 0.0,
        },
        "workers": [
            {"worker_id": "qwen", "pass_rate": 1.0, "mean_latency_ms": 10.0},
            {"worker_id": "kimi", "pass_rate": 1.0, "mean_latency_ms": 20.0},
        ],
    }


class OrchestratorRuntimeTest(unittest.TestCase):
    def test_loads_model_payload_and_predicts_record(self) -> None:
        records = [
            example("t1", "filesystem", "qwen"),
            example("t2", "network", "kimi"),
        ]
        model, _ = train_multi_head_orchestrator(records, epochs=80, learning_rate=0.05)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            model_path.write_text(
                json.dumps(
                    {
                        "model_type": "linear-multi-head-orchestrator",
                        "substrate": "fixture.jsonl",
                        "orchestrator": model.to_dict(),
                    }
                ),
                encoding="utf-8",
            )

            restored, payload = load_multi_head_orchestrator(model_path)
            prediction = predict_orchestration(model_path=model_path, record=records[0])

        self.assertEqual(payload["model_type"], "linear-multi-head-orchestrator")
        self.assertEqual(restored.worker_ids, ["kimi", "qwen"])
        self.assertEqual(prediction["selected_worker_id"], "qwen")
        self.assertIn("worker_distribution", prediction)
        self.assertEqual(prediction["substrate"], "fixture.jsonl")

    def test_build_prompt_record_derives_dense_features(self) -> None:
        record = build_prompt_record(
            prompt="Read files and summarize sorted counts.",
            task_id="adhoc-1",
            categories=["filesystem", "text"],
            libraries=["pathlib"],
        )

        self.assertEqual(record["task_id"], "adhoc-1")
        self.assertEqual(record["dense_features"]["signal_filesystem"], 1.0)
        self.assertEqual(record["dense_features"]["lib_pathlib"], 1.0)

    def test_normalize_record_requires_prompt_or_dense_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "dense_features or prompt"):
            normalize_prediction_record({"task_id": "broken"})


if __name__ == "__main__":
    unittest.main()
