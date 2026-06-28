import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mempool.router_baseline import (
    FamilyRouter,
    NearestNeighborRouter,
    evaluate_policy,
    strongest_worker,
)


class RouterBaselineTest(unittest.TestCase):
    def test_family_router_memorizes_family_targets(self) -> None:
        records = [
            {
                "task_family": "a",
                "target_worker_id": "w1",
                "workers": [
                    {"worker_id": "w1", "passed": True, "latency_ms": 10, "cost_usd": 0.0},
                    {"worker_id": "w2", "passed": False, "latency_ms": 1, "cost_usd": 0.0},
                ],
            },
            {
                "task_family": "b",
                "target_worker_id": "w2",
                "workers": [
                    {"worker_id": "w1", "passed": True, "latency_ms": 10, "cost_usd": 0.0},
                    {"worker_id": "w2", "passed": True, "latency_ms": 1, "cost_usd": 0.0},
                ],
            },
        ]

        router = FamilyRouter.train(records)
        predictions = [router.predict(record) for record in records]
        evaluation = evaluate_policy(records, "family-router", predictions)

        self.assertEqual(predictions, ["w1", "w2"])
        self.assertEqual(evaluation.target_accuracy, 1.0)
        self.assertEqual(evaluation.pass_at_1, 1.0)

    def test_nearest_neighbor_predicts_closest_target(self) -> None:
        records = [
            {
                "task_id": "a",
                "prompt": "sum two numbers",
                "task_family": "code_easy",
                "target_worker_id": "w1",
            },
            {
                "task_id": "b",
                "prompt": "lowercase dict keys",
                "task_family": "code_data",
                "target_worker_id": "w2",
            },
        ]

        router = NearestNeighborRouter.train(records)
        prediction = router.predict(
            {
                "task_id": "c",
                "prompt": "strip dict keys",
                "task_family": "code_data",
                "target_worker_id": "w2",
            }
        )

        self.assertEqual(prediction, "w2")

    def test_strongest_worker_tie_breaks_by_latency(self) -> None:
        records = [
            {
                "workers": [
                    {"worker_id": "slow", "passed": True, "latency_ms": 100, "cost_usd": 0.0},
                    {"worker_id": "fast", "passed": True, "latency_ms": 10, "cost_usd": 0.0},
                ],
            },
            {
                "workers": [
                    {"worker_id": "slow", "passed": False, "latency_ms": 100, "cost_usd": 0.0},
                    {"worker_id": "fast", "passed": False, "latency_ms": 10, "cost_usd": 0.0},
                ],
            },
        ]

        self.assertEqual(strongest_worker(records), "fast")

    def test_evaluate_policy_reports_latency_regret_against_target(self) -> None:
        records = [
            {
                "task_family": "a",
                "target_worker_id": "fast",
                "workers": [
                    {"worker_id": "fast", "passed": True, "latency_ms": 10, "cost_usd": 0.0},
                    {"worker_id": "slow", "passed": True, "latency_ms": 40, "cost_usd": 0.0},
                ],
            },
            {
                "task_family": "a",
                "target_worker_id": "fast",
                "workers": [
                    {"worker_id": "fast", "passed": True, "latency_ms": 20, "cost_usd": 0.0},
                    {"worker_id": "slow", "passed": True, "latency_ms": 50, "cost_usd": 0.0},
                ],
            },
        ]

        evaluation = evaluate_policy(records, "slow-policy", ["slow", "slow"])

        self.assertEqual(evaluation.mean_latency_ms, 45.0)
        self.assertEqual(evaluation.mean_target_latency_ms, 15.0)
        self.assertEqual(evaluation.mean_latency_regret_ms, 30.0)

    def test_evaluate_policy_reports_solvable_subset(self) -> None:
        records = [
            {
                "task_family": "a",
                "target_worker_id": "fast-fail",
                "workers": [
                    {"worker_id": "fast-fail", "passed": False, "latency_ms": 1, "cost_usd": 0.0},
                    {"worker_id": "slow-fail", "passed": False, "latency_ms": 10, "cost_usd": 0.0},
                ],
            },
            {
                "task_family": "a",
                "target_worker_id": "solver",
                "workers": [
                    {"worker_id": "fast-fail", "passed": False, "latency_ms": 1, "cost_usd": 0.0},
                    {"worker_id": "solver", "passed": True, "latency_ms": 20, "cost_usd": 0.0},
                ],
            },
        ]

        evaluation = evaluate_policy(records, "fast-policy", ["fast-fail", "fast-fail"])

        self.assertEqual(evaluation.target_accuracy, 0.5)
        self.assertEqual(evaluation.pass_at_1, 0.0)
        self.assertEqual(evaluation.solvable_task_count, 1)
        self.assertEqual(evaluation.solvable_pass_at_1, 0.0)
        self.assertEqual(evaluation.solvable_target_accuracy, 0.0)

    def test_train_cli_reports_unavailable_loo_for_single_record(self) -> None:
        record = {
            "task_id": "task-1",
            "benchmark_id": "external",
            "task_family": "code",
            "prompt": "solve it",
            "prompt_features": {"length_chars": 8},
            "target_worker_id": "w1",
            "target_distribution": {"w1": 1.0},
            "workers": [
                {
                    "worker_id": "w1",
                    "model": "model",
                    "passed": False,
                    "score": 0.0,
                    "latency_ms": 10,
                    "cost_usd": 0.0,
                    "failure_mode": "test_failure",
                    "reward": -0.05,
                    "target_probability": 1.0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "dataset.jsonl"
            output = Path(tmpdir) / "report.json"
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/train_router_baseline.py",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))

        loo = [item for item in report["evaluations"] if item["policy"] == "family-router-loo"][0]
        self.assertFalse(loo["available"])

    def test_train_cli_includes_active_policy_when_registry_is_supplied(self) -> None:
        record = {
            "task_id": "task-1",
            "benchmark_id": "external",
            "task_family": "code",
            "prompt": "solve it",
            "prompt_features": {"length_chars": 8},
            "target_worker_id": "w1",
            "target_distribution": {"w1": 1.0},
            "workers": [
                {
                    "worker_id": "w1",
                    "model": "model",
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
        model = {
            "router": {
                "worker_ids": ["w1"],
                "feature_names": ["bias"],
                "weights": [[0.0]],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.jsonl"
            output = root / "report.json"
            model_path = root / "model.json"
            registry = root / "registry.json"
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
            model_path.write_text(json.dumps(model), encoding="utf-8")
            registry.write_text(
                json.dumps({"active": {"model": str(model_path), "dataset": str(dataset)}}),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/train_router_baseline.py",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--active-policy-registry",
                    str(registry),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))

        active = [item for item in report["evaluations"] if item["policy"] == "active-logits-router"]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["target_accuracy"], 1.0)
        self.assertEqual(report["active_policy"]["model"], str(model_path))

    def test_train_cli_includes_probe_gated_policy_when_artifact_is_supplied(self) -> None:
        record = {
            "task_id": "task-1",
            "benchmark_id": "external",
            "task_family": "code",
            "prompt": "solve it",
            "prompt_features": {"length_chars": 8},
            "target_worker_id": "fast",
            "target_distribution": {"slow": 0.55, "fast": 0.45},
            "workers": [
                {
                    "worker_id": "slow",
                    "model": "slow",
                    "passed": True,
                    "pass_rate": 1.0,
                    "score": 1.0,
                    "latency_ms": 10000,
                    "cost_usd": 0.0,
                    "failure_mode": None,
                    "reward": 0.55,
                    "target_probability": 0.55,
                },
                {
                    "worker_id": "fast",
                    "model": "fast",
                    "passed": True,
                    "pass_rate": 1.0,
                    "score": 1.0,
                    "latency_ms": 1000,
                    "cost_usd": 0.0,
                    "failure_mode": None,
                    "reward": 0.45,
                    "target_probability": 0.45,
                },
            ],
        }
        model = {
            "router": {
                "worker_ids": ["slow", "fast"],
                "feature_names": ["bias"],
                "weights": [[0.55], [0.45]],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.jsonl"
            output = root / "report.json"
            model_path = root / "model.json"
            policy_path = root / "probe-policy.json"
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
            model_path.write_text(json.dumps(model), encoding="utf-8")
            policy_path.write_text(
                json.dumps(
                    {
                        "policy": "probe-gated-latency-calibrated-logits-router",
                        "base_model": str(model_path),
                        "probe_gate": {
                            "probe_worker_ids": ["slow"],
                            "mode": "all",
                            "min_pass_rate": 1.0,
                        },
                        "calibration": {
                            "latency_cost_per_second": 0.5,
                            "min_probability_ratio": 0.0,
                            "min_probability": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/train_router_baseline.py",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--probe-gated-policy",
                    str(policy_path),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))

        probe_gated = [
            item
            for item in report["evaluations"]
            if item["policy"] == "probe-gated-latency-calibrated-logits-router"
        ]
        self.assertEqual(len(probe_gated), 1)
        self.assertEqual(probe_gated[0]["target_accuracy"], 1.0)
        self.assertEqual(probe_gated[0]["changed_from_top"], 1)
        self.assertEqual(report["probe_gated_policy"]["artifact"], str(policy_path))


if __name__ == "__main__":
    unittest.main()
