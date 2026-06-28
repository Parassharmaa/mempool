import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_active_policy import (
    evaluate_active_policy_payload,
    load_active_policy,
    load_active_router,
)


class EvaluateActivePolicyTest(unittest.TestCase):
    def test_loads_active_router_from_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.json"
            registry = Path(tmpdir) / "registry.json"
            model.write_text(
                json.dumps(
                    {
                        "router": {
                            "worker_ids": ["w1"],
                            "feature_names": ["bias"],
                            "weights": [[0.0]],
                        }
                    }
                ),
                encoding="utf-8",
            )
            registry.write_text(
                json.dumps({"active": {"model": str(model), "dataset": "dataset.jsonl"}}),
                encoding="utf-8",
            )

            router, active = load_active_router(registry)

        self.assertEqual(router.worker_ids, ["w1"])
        self.assertEqual(active["dataset"], "dataset.jsonl")

    def test_rejects_missing_active_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = Path(tmpdir) / "registry.json"
            registry.write_text(json.dumps({"active": None}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_active_router(registry)

    def test_generic_loader_evaluates_logits_router(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "model.json"
            dataset = root / "routing.jsonl"
            registry = root / "registry.json"
            model.write_text(
                json.dumps(
                    {
                        "model_type": "logits-router",
                        "router": {
                            "worker_ids": ["w1"],
                            "feature_names": ["bias"],
                            "weights": [[0.0]],
                        },
                    }
                ),
                encoding="utf-8",
            )
            dataset.write_text(
                json.dumps(
                    {
                        "task_id": "t1",
                        "benchmark_id": "b",
                        "task_family": "unit",
                        "prompt": "do it",
                        "prompt_features": {},
                        "target_worker_id": "w1",
                        "target_distribution": {"w1": 1.0},
                        "workers": [
                            {
                                "worker_id": "w1",
                                "model": "m",
                                "passed": True,
                                "score": 1.0,
                                "latency_ms": 10,
                                "cost_usd": 0.0,
                                "failure_mode": None,
                                "reward": 1.0,
                                "target_probability": 1.0,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            registry.write_text(
                json.dumps({"active": {"model": str(model), "dataset": str(dataset)}}),
                encoding="utf-8",
            )

            payload, active = load_active_policy(registry)
            evaluation, evaluated_dataset = evaluate_active_policy_payload(payload, active)

        self.assertEqual(evaluated_dataset, dataset)
        self.assertEqual(evaluation["policy"], "logits-router")
        self.assertEqual(evaluation["pass_at_1"], 1.0)

    def test_generic_loader_accepts_linear_softmax_logits_router_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "model.json"
            dataset = root / "routing.jsonl"
            registry = root / "registry.json"
            model.write_text(
                json.dumps(
                    {
                        "model_type": "linear-softmax-logits-router",
                        "router": {
                            "worker_ids": ["w1"],
                            "feature_names": ["bias"],
                            "weights": [[0.0]],
                        },
                    }
                ),
                encoding="utf-8",
            )
            dataset.write_text(
                json.dumps(
                    {
                        "task_id": "t1",
                        "benchmark_id": "b",
                        "task_family": "unit",
                        "prompt": "do it",
                        "prompt_features": {},
                        "target_worker_id": "w1",
                        "target_distribution": {"w1": 1.0},
                        "workers": [
                            {
                                "worker_id": "w1",
                                "model": "m",
                                "passed": True,
                                "score": 1.0,
                                "latency_ms": 10,
                                "cost_usd": 0.0,
                                "failure_mode": None,
                                "reward": 1.0,
                                "target_probability": 1.0,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            registry.write_text(
                json.dumps({"active": {"model": str(model), "dataset": str(dataset)}}),
                encoding="utf-8",
            )

            router, _ = load_active_router(registry)
            payload, active = load_active_policy(registry)
            evaluation, _ = evaluate_active_policy_payload(payload, active)

        self.assertEqual(router.worker_ids, ["w1"])
        self.assertEqual(evaluation["pass_at_1"], 1.0)

    def test_generic_loader_evaluates_latency_calibrated_multi_head_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_model = root / "multihead.json"
            calibrated_model = root / "calibrated.json"
            substrate = root / "substrate.jsonl"
            registry = root / "registry.json"
            base_model.write_text(
                json.dumps(
                    {
                        "model_type": "linear-multi-head-orchestrator",
                        "orchestrator": {
                            "worker_ids": ["slow", "fast"],
                            "workflow_labels": ["direct", "verify_then_fallback"],
                            "feature_names": ["bias"],
                            "worker_weights": [[0.0], [0.0]],
                            "workflow_weights": [[0.0], [0.0]],
                            "verifier_weights": [0.0],
                            "abstain_weights": [0.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            calibrated_model.write_text(
                json.dumps(
                    {
                        "model_type": "latency-calibrated-multi-head-router",
                        "base_model": str(base_model),
                        "substrate": str(substrate),
                        "calibration": {
                            "policy": "latency-calibrated-worker-choice",
                            "latency_cost_per_second": 0.01,
                            "min_probability_ratio": 0.0,
                            "min_probability": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            substrate.write_text(
                json.dumps(
                    {
                        "task_id": "t1",
                        "dense_features": {"bias": 1.0},
                        "target": {
                            "worker_distribution": {"slow": 0.5, "fast": 0.5},
                            "target_worker_id": "fast",
                            "workflow_distribution": {"direct": 1.0, "verify_then_fallback": 0.0},
                            "workflow_kind": "direct",
                            "verifier_probability": 0.0,
                            "abstain_probability": 0.0,
                        },
                        "workers": [
                            {"worker_id": "slow", "pass_rate": 1.0, "mean_latency_ms": 10000.0},
                            {"worker_id": "fast", "pass_rate": 1.0, "mean_latency_ms": 1000.0},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            registry.write_text(
                json.dumps(
                    {"active": {"model": str(calibrated_model), "dataset": "routing.jsonl"}}
                ),
                encoding="utf-8",
            )

            payload, active = load_active_policy(registry)
            evaluation, evaluated_dataset = evaluate_active_policy_payload(payload, active)

        self.assertEqual(evaluated_dataset, substrate)
        self.assertEqual(evaluation["policy"], "latency-calibrated-worker-choice")
        self.assertEqual(evaluation["target_accuracy"], 1.0)
        self.assertEqual(evaluation["mean_latency_regret_ms"], 0.0)

    def test_router_only_loader_rejects_calibrated_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "calibrated.json"
            registry = root / "registry.json"
            model.write_text(
                json.dumps(
                    {
                        "model_type": "latency-calibrated-multi-head-router",
                        "base_model": "base.json",
                        "substrate": "substrate.jsonl",
                        "calibration": {},
                    }
                ),
                encoding="utf-8",
            )
            registry.write_text(
                json.dumps({"active": {"model": str(model), "dataset": "dataset.jsonl"}}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_active_router(registry)


if __name__ == "__main__":
    unittest.main()
