import tempfile
import unittest
from pathlib import Path

from tools.gate_operational_policy import evaluation_metrics, gate_operational_policy


class GateOperationalPolicyTest(unittest.TestCase):
    def test_reads_top_level_evaluation(self) -> None:
        metrics = evaluation_metrics(
            {
                "evaluation": {
                    "policy": "gated-fallback",
                    "target_accuracy": 0.7,
                    "pass_at_1": 0.8,
                    "solvable_task_count": 3,
                    "solvable_pass_at_1": 0.9,
                    "solvable_target_accuracy": 0.6,
                    "mean_latency_regret_ms": 100.0,
                    "mean_latency_ms": 500.0,
                    "task_count": 4,
                }
            }
        )

        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["policy"], "gated-fallback")
        self.assertEqual(metrics["pass_at_1"], 0.8)

    def test_promotes_candidate_that_matches_reference_and_strongest_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "routing.jsonl"
            dataset.write_text(
                "\n".join(
                    [
                        '{"task_id":"t0","target_worker_id":"a","workers":[{"worker_id":"a","passed":true},{"worker_id":"b","passed":false}]}',
                        '{"task_id":"t1","target_worker_id":"b","workers":[{"worker_id":"a","passed":false},{"worker_id":"b","passed":true}]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = gate_operational_policy(
                {
                    "evaluation": {
                        "policy": "candidate",
                        "target_accuracy": 1.0,
                        "pass_at_1": 1.0,
                        "solvable_task_count": 2,
                        "solvable_pass_at_1": 1.0,
                        "solvable_target_accuracy": 1.0,
                        "mean_latency_regret_ms": 0.0,
                        "task_count": 2,
                    }
                },
                {
                    "evaluations": [
                        {
                            "policy": "reference",
                            "target_accuracy": 1.0,
                            "pass_at_1": 1.0,
                            "solvable_task_count": 2,
                            "solvable_pass_at_1": 1.0,
                            "solvable_target_accuracy": 1.0,
                            "mean_latency_regret_ms": 0.0,
                            "task_count": 2,
                        }
                    ]
                },
                reference_policy="reference",
                dataset=dataset,
                min_pass_at_1_vs_strongest=0.0,
            )

        self.assertEqual(result["decision"], "promote")
        self.assertEqual(result["strongest_worker"]["pass_at_1"], 0.5)

    def test_quarantines_candidate_with_latency_regret_increase(self) -> None:
        result = gate_operational_policy(
            {
                "evaluation": {
                    "policy": "candidate",
                    "target_accuracy": 0.9,
                    "pass_at_1": 0.9,
                    "solvable_task_count": 10,
                    "solvable_pass_at_1": 0.9,
                    "solvable_target_accuracy": 0.9,
                    "mean_latency_regret_ms": 400.0,
                    "task_count": 10,
                }
            },
            {
                "evaluations": [
                    {
                        "policy": "reference",
                        "target_accuracy": 0.9,
                        "pass_at_1": 0.9,
                        "solvable_task_count": 10,
                        "solvable_pass_at_1": 0.9,
                        "solvable_target_accuracy": 0.9,
                        "mean_latency_regret_ms": 100.0,
                        "task_count": 10,
                    }
                ]
            },
            reference_policy="reference",
            max_latency_regret_increase_ms=0.0,
        )

        self.assertEqual(result["decision"], "quarantine")
        self.assertIn("latency regret increase", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
