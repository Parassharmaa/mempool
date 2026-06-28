import json
import tempfile
import unittest
from pathlib import Path

from mempool.small_orchestrator_substrate import build_orchestrator_substrate


def record(task_id: str = "t1") -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "bigcodebench_hard",
        "prompt": "Use pandas to solve the task.",
        "prompt_features": {
            "categories": ["datasci"],
            "libraries": ["pandas"],
            "missing_libraries": [],
        },
        "target_worker_id": "w1",
        "target_distribution": {"w1": 0.9, "w2": 0.1},
        "workers": [
            {
                "worker_id": "w1",
                "model": "m1",
                "passed": True,
                "score": 1.0,
                "latency_ms": 10,
                "cost_usd": 0.0,
                "failure_mode": None,
                "reward": 0.9,
                "target_probability": 0.9,
            },
            {
                "worker_id": "w2",
                "model": "m2",
                "passed": True,
                "score": 1.0,
                "latency_ms": 20,
                "cost_usd": 0.0,
                "failure_mode": None,
                "reward": 0.1,
                "target_probability": 0.1,
            },
        ],
    }


def contract() -> dict:
    return {
        "heads": {
            "worker_distribution": {
                "head_type": "softmax",
                "worker_ids": ["w1", "w2"],
                "weights": [[0.0], [0.0]],
            },
            "workflow_kind": {
                "head_type": "softmax",
                "labels": ["direct", "verify_then_fallback"],
                "logits": [1.0, 0.0],
            },
            "verifier_probability": {
                "head_type": "sigmoid",
                "logit": 0.0,
                "threshold": 0.5,
            },
            "abstain_probability": {
                "head_type": "sigmoid",
                "logit": -3.0,
                "threshold": 0.5,
            },
        }
    }


class SmallOrchestratorSubstrateTest(unittest.TestCase):
    def test_exports_multi_head_examples_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "routing.jsonl"
            dataset.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract()), encoding="utf-8")
            output = root / "substrate.jsonl"
            manifest = root / "manifest.json"

            payload = build_orchestrator_substrate(
                routing_dataset_path=dataset,
                contract_path=contract_path,
                output_path=output,
                manifest_path=manifest,
                reward_temperature=0.05,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(payload["record_count"], 1)
        self.assertEqual(payload["target_counts"], {"w1": 1})
        self.assertEqual(payload["workflow_counts"], {"direct": 1})
        self.assertEqual(payload["abstain_positive"], 0)
        self.assertEqual(rows[0]["target"]["target_worker_id"], "w1")
        self.assertEqual(rows[0]["target"]["workflow_kind"], "direct")
        self.assertIn("worker_distribution", rows[0]["target"])
        self.assertEqual(rows[0]["messages"][-1]["role"], "assistant")
        self.assertEqual(json.loads(rows[0]["messages"][-1]["content"]), rows[0]["target"])


if __name__ == "__main__":
    unittest.main()
