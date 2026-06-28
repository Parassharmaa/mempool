import json
import tempfile
import unittest
from pathlib import Path

from tools.select_multi_head_fallback_threshold import (
    policy_payload,
    select_multi_head_fallback_threshold,
)


def record() -> dict:
    return {
        "task_id": "t1",
        "target": {
            "worker_distribution": {"a": 0.2, "b": 0.8},
            "target_worker_id": "b",
            "workflow_distribution": {"direct": 1.0, "verify_then_fallback": 0.0},
            "workflow_kind": "direct",
            "verifier_probability": 1.0,
            "abstain_probability": 0.0,
        },
        "dense_features": {"bias": 1.0},
        "workers": [
            {"worker_id": "a", "pass_rate": 0.0, "mean_latency_ms": 1.0},
            {"worker_id": "b", "pass_rate": 1.0, "mean_latency_ms": 5.0},
        ],
    }


class SelectMultiHeadFallbackThresholdTest(unittest.TestCase):
    def test_selects_threshold_from_report_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            substrate = root / "substrate.jsonl"
            substrate.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "model_output": "model.json",
                        "leave_one_out": {
                            "target_accuracy": 0.0,
                            "pass_at_1": 0.0,
                            "solvable_pass_at_1": 0.0,
                            "mean_latency_regret_ms": 1.0,
                            "predictions": [
                                {
                                    "task_id": "t1",
                                    "worker_distribution": {"a": 0.52, "b": 0.48},
                                    "verifier_probability": 0.9,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            selection = select_multi_head_fallback_threshold(
                substrate_path=substrate,
                report_path=report,
                margins=[0.1],
                verifier_thresholds=[0.5],
            )
            policy = policy_payload(selection)

        self.assertTrue(selection["valid"])
        self.assertEqual(selection["selected"]["evaluation"]["solved"], 1)
        self.assertEqual(selection["lowest_regret_pass_gain"]["evaluation"]["solved"], 1)
        self.assertEqual(policy["policy"], "multi-head-gated-fallback")

    def test_rejects_reports_without_worker_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            substrate = root / "substrate.jsonl"
            substrate.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            report = root / "report.json"
            report.write_text(
                json.dumps({"leave_one_out": {"predictions": [{"task_id": "t1"}]}}),
                encoding="utf-8",
            )

            selection = select_multi_head_fallback_threshold(
                substrate_path=substrate,
                report_path=report,
                margins=[0.1],
                verifier_thresholds=[0.5],
            )

        self.assertFalse(selection["valid"])
        self.assertIn("worker_distribution", selection["errors"][0])


if __name__ == "__main__":
    unittest.main()
