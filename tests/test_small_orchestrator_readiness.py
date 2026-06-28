import json
import tempfile
import unittest
from pathlib import Path

from mempool.small_orchestrator_readiness import audit_small_orchestrator_readiness


def routing_record(task_id: str, target: str = "w1") -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "code",
        "prompt": "solve",
        "prompt_features": {"libraries": []},
        "target_worker_id": target,
        "target_distribution": {"w1": 0.7, "w2": 0.3},
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
                "target_probability": 0.7,
            },
            {
                "worker_id": "w2",
                "model": "m2",
                "passed": False,
                "score": 0.0,
                "latency_ms": 20,
                "cost_usd": 0.0,
                "failure_mode": "failed",
                "reward": 0.0,
                "target_probability": 0.3,
            },
        ],
    }


class SmallOrchestratorReadinessTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_jsonl(self, path: Path, rows: list[dict]) -> Path:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_reports_missing_m5_action_heads_and_data_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = self.write_jsonl(root / "dataset.jsonl", [routing_record("t1")])
            model = self.write_json(
                root / "model.json",
                {
                    "model_type": "linear-softmax-logits-router",
                    "router": {
                        "policy": "logits-router",
                        "worker_ids": ["w1", "w2"],
                        "feature_names": ["bias"],
                        "weights": [[0.0], [0.0]],
                    },
                },
            )
            registry = self.write_json(
                root / "registry.json",
                {
                    "active": {
                        "model": str(model),
                        "dataset": str(dataset),
                        "loo": {
                            "available": True,
                            "task_count": 1,
                            "target_accuracy": 1.0,
                            "solvable_pass_at_1": 1.0,
                            "mean_latency_regret_ms": 0.0,
                        },
                        "target_mix": {
                            "task_count": 1,
                            "target_worker_count": 1,
                        },
                    }
                },
            )
            fallback = self.write_json(
                root / "fallback.json",
                {"evaluation": {"policy": "gated", "solvable_pass_at_1": 1.0}},
            )
            regression = self.write_json(root / "regression.json", {"passed": True})
            contract = self.write_json(root / "contract.json", {"heads": {}})

            report = audit_small_orchestrator_readiness(
                registry_path=registry,
                fallback_report_path=fallback,
                regression_report_path=regression,
                orchestrator_contract_path=contract,
                min_tasks=2,
                min_target_workers=2,
                min_workers_per_task=2,
            )

        self.assertFalse(report["ready_for_m5_small_orchestrator"])
        self.assertEqual(report["decision"], "continue-m3-data-and-heads")
        self.assertTrue(any("active dataset has 1 tasks" in reason for reason in report["reasons"]))
        self.assertTrue(any("workflow-kind logits head" in reason for reason in report["reasons"]))
        self.assertTrue(any("abstain probability head" in reason for reason in report["reasons"]))

    def test_can_pass_when_relaxed_action_head_requirements_are_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = self.write_jsonl(
                root / "dataset.jsonl",
                [routing_record("t1"), routing_record("t2", target="w2")],
            )
            model = self.write_json(
                root / "model.json",
                {
                    "model_type": "linear-softmax-logits-router",
                    "router": {
                        "policy": "logits-router",
                        "worker_ids": ["w1", "w2"],
                        "feature_names": ["bias"],
                        "weights": [[0.0], [0.0]],
                    },
                },
            )
            registry = self.write_json(
                root / "registry.json",
                {
                    "active": {
                        "model": str(model),
                        "dataset": str(dataset),
                        "loo": {
                            "available": True,
                            "task_count": 2,
                            "target_accuracy": 0.9,
                            "solvable_pass_at_1": 1.0,
                            "mean_latency_regret_ms": 5.0,
                        },
                        "target_mix": {
                            "task_count": 2,
                            "target_worker_count": 2,
                        },
                    }
                },
            )
            fallback = self.write_json(
                root / "fallback.json",
                {"evaluation": {"policy": "gated", "solvable_pass_at_1": 1.0}},
            )
            regression = self.write_json(root / "regression.json", {"passed": True})
            contract = self.write_json(
                root / "contract.json",
                {
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
                },
            )

            report = audit_small_orchestrator_readiness(
                registry_path=registry,
                fallback_report_path=fallback,
                regression_report_path=regression,
                orchestrator_contract_path=contract,
                min_tasks=2,
                min_target_workers=2,
                min_workers_per_task=2,
            )

        self.assertTrue(report["ready_for_m5_small_orchestrator"])
        self.assertEqual(report["decision"], "start-m5")


if __name__ == "__main__":
    unittest.main()
