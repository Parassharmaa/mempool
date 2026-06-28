import tempfile
import unittest
from pathlib import Path

from tools.policy_refresh_gate import (
    evaluate_refresh,
    policy_evaluation_metrics,
    thresholds_for_profile,
)


def report(
    acc: float,
    tasks: int = 2,
    latency_regret_ms: float = 0.0,
    solvable_tasks: int = 0,
    solvable_pass_at_1: float = 0.0,
) -> dict:
    return {
        "model_output": "model.json",
        "leave_one_out": {
            "available": True,
            "target_accuracy": acc,
            "pass_at_1": acc,
            "solvable_task_count": solvable_tasks,
            "solvable_pass_at_1": solvable_pass_at_1,
            "solvable_target_accuracy": solvable_pass_at_1,
            "mean_kl": 0.1,
            "mean_latency_regret_ms": latency_regret_ms,
            "task_count": tasks,
        },
    }


class PolicyRefreshGateTest(unittest.TestCase):
    def write_dataset(self, tmpdir: str, name: str, targets: list[str]) -> Path:
        path = Path(tmpdir) / name
        path.write_text(
            "".join(
                f'{{"task_id":"t{index}","target_worker_id":"{target}"}}\n'
                for index, target in enumerate(targets)
            ),
            encoding="utf-8",
        )
        return path

    def write_worker_dataset(self, tmpdir: str, name: str) -> Path:
        path = Path(tmpdir) / name
        path.write_text(
            "\n".join(
                [
                    '{"task_id":"t0","target_worker_id":"strong","workers":[{"worker_id":"strong","passed":true},{"worker_id":"weak","passed":false}]}',
                    '{"task_id":"t1","target_worker_id":"strong","workers":[{"worker_id":"strong","passed":true},{"worker_id":"weak","passed":false}]}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_promotes_richer_candidate_with_allowed_loo_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = self.write_dataset(tmpdir, "baseline.jsonl", ["qwen", "kimi"])
            candidate = self.write_dataset(tmpdir, "candidate.jsonl", ["qwen", "kimi", "glm"])

            result = evaluate_refresh(
                candidate_report=report(0.75, tasks=3),
                candidate_dataset=candidate,
                baseline_report=report(0.83, tasks=2),
                baseline_dataset=baseline,
            )

        self.assertEqual(result["decision"], "promote")
        self.assertEqual(result["candidate"]["target_mix"]["target_worker_count"], 3)
        self.assertTrue(result["warnings"])

    def test_quarantines_large_loo_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = self.write_dataset(tmpdir, "baseline.jsonl", ["qwen", "kimi"])
            candidate = self.write_dataset(tmpdir, "candidate.jsonl", ["qwen", "kimi", "glm"])

            result = evaluate_refresh(
                candidate_report=report(0.5, tasks=3),
                candidate_dataset=candidate,
                baseline_report=report(0.83, tasks=2),
                baseline_dataset=baseline,
            )

        self.assertEqual(result["decision"], "quarantine")
        self.assertTrue(result["reasons"])

    def test_allows_accuracy_drop_equal_to_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = self.write_dataset(tmpdir, "baseline.jsonl", ["qwen", "kimi"])
            candidate = self.write_dataset(tmpdir, "candidate.jsonl", ["qwen", "kimi"])

            result = evaluate_refresh(
                candidate_report=report(0.7, tasks=2),
                candidate_dataset=candidate,
                baseline_report=report(0.8, tasks=2),
                baseline_dataset=baseline,
                max_loo_accuracy_drop=0.1,
            )

        self.assertEqual(result["decision"], "promote")
        self.assertTrue(result["warnings"])

    def test_quarantines_high_latency_regret(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = self.write_dataset(tmpdir, "candidate.jsonl", ["qwen", "kimi"])

            result = evaluate_refresh(
                candidate_report=report(0.9, tasks=2, latency_regret_ms=1500.0),
                candidate_dataset=candidate,
                max_loo_latency_regret_ms=1000.0,
            )

        self.assertEqual(result["decision"], "quarantine")
        self.assertIn("latency regret", result["reasons"][0])

    def test_quarantines_latency_regret_increase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = self.write_dataset(tmpdir, "baseline.jsonl", ["qwen", "kimi"])
            candidate = self.write_dataset(tmpdir, "candidate.jsonl", ["qwen", "kimi", "glm"])

            result = evaluate_refresh(
                candidate_report=report(0.9, tasks=3, latency_regret_ms=900.0),
                candidate_dataset=candidate,
                baseline_report=report(0.9, tasks=2, latency_regret_ms=100.0),
                baseline_dataset=baseline,
                max_loo_latency_regret_increase_ms=250.0,
            )

        self.assertEqual(result["decision"], "quarantine")
        self.assertIn("latency regret increase", result["reasons"][0])

    def test_quarantines_low_solvable_pass_rate_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = self.write_dataset(tmpdir, "candidate.jsonl", ["qwen", "deepseek"])

            result = evaluate_refresh(
                candidate_report=report(
                    0.8,
                    tasks=2,
                    solvable_tasks=1,
                    solvable_pass_at_1=0.0,
                ),
                candidate_dataset=candidate,
                min_loo_solvable_pass_at_1=0.5,
            )

        self.assertEqual(result["decision"], "quarantine")
        self.assertIn("solvable pass@1", result["reasons"][0])

    def test_quarantines_candidate_below_strongest_worker_pass_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = self.write_worker_dataset(tmpdir, "candidate.jsonl")

            result = evaluate_refresh(
                candidate_report=report(0.8, tasks=2),
                candidate_dataset=candidate,
                min_loo_accuracy=0.0,
                min_loo_pass_at_1_vs_strongest=0.0,
            )

        self.assertEqual(result["decision"], "quarantine")
        self.assertIn("strongest-worker pass@1", result["reasons"][0])
        self.assertEqual(result["strongest_worker"]["pass_at_1"], 1.0)

    def test_preserve_accuracy_profile_derives_baseline_thresholds(self) -> None:
        thresholds = thresholds_for_profile(
            profile="preserve_accuracy",
            baseline_report=report(
                0.82,
                tasks=4,
                latency_regret_ms=450.0,
                solvable_tasks=3,
                solvable_pass_at_1=0.75,
            ),
            min_loo_accuracy=0.7,
            max_loo_accuracy_drop=0.1,
            min_loo_solvable_pass_at_1=None,
            max_loo_latency_regret_ms=None,
            max_loo_latency_regret_increase_ms=None,
            min_loo_pass_at_1_vs_strongest=None,
        )

        self.assertEqual(thresholds["min_loo_accuracy"], 0.82)
        self.assertEqual(thresholds["max_loo_accuracy_drop"], 0.0)
        self.assertEqual(thresholds["min_loo_solvable_pass_at_1"], 0.75)
        self.assertEqual(thresholds["max_loo_latency_regret_ms"], 450.0)

    def test_extracts_named_operational_policy_metrics(self) -> None:
        metrics = policy_evaluation_metrics(
            {
                "evaluations": [
                    {
                        "policy": "active-logits-router",
                        "target_accuracy": 0.7,
                        "pass_at_1": 0.8,
                        "solvable_task_count": 3,
                        "solvable_pass_at_1": 0.9,
                        "solvable_target_accuracy": 0.6,
                        "mean_latency_regret_ms": 250.0,
                        "task_count": 4,
                    }
                ]
            },
            "active-logits-router",
        )

        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["target_accuracy"], 0.7)
        self.assertEqual(metrics["pass_at_1"], 0.8)
        self.assertEqual(metrics["mean_latency_regret_ms"], 250.0)

    def test_preserve_accuracy_profile_uses_operational_reference(self) -> None:
        operational = {
            "available": True,
            "policy": "probe-gated",
            "target_accuracy": 0.83,
            "pass_at_1": 0.77,
            "solvable_task_count": 5,
            "solvable_pass_at_1": 0.86,
            "solvable_target_accuracy": 0.81,
            "mean_latency_regret_ms": 251.0,
            "task_count": 6,
        }
        thresholds = thresholds_for_profile(
            profile="preserve_accuracy",
            baseline_report=report(0.78, tasks=6, latency_regret_ms=501.0),
            operational_reference=operational,
            min_loo_accuracy=0.7,
            max_loo_accuracy_drop=0.1,
            min_loo_solvable_pass_at_1=None,
            max_loo_latency_regret_ms=None,
            max_loo_latency_regret_increase_ms=None,
            min_loo_pass_at_1_vs_strongest=None,
        )

        self.assertEqual(thresholds["min_loo_accuracy"], 0.83)
        self.assertEqual(thresholds["max_loo_accuracy_drop"], 0.0)
        self.assertEqual(thresholds["min_loo_solvable_pass_at_1"], 0.86)
        self.assertEqual(thresholds["max_loo_latency_regret_ms"], 251.0)

    def test_quarantines_drop_against_operational_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = self.write_dataset(tmpdir, "candidate.jsonl", ["qwen", "kimi"])

            result = evaluate_refresh(
                candidate_report=report(0.8, tasks=2, latency_regret_ms=300.0),
                candidate_dataset=candidate,
                operational_reference={
                    "available": True,
                    "policy": "probe-gated",
                    "target_accuracy": 0.83,
                    "pass_at_1": 0.77,
                    "solvable_task_count": 0,
                    "solvable_pass_at_1": 0.0,
                    "solvable_target_accuracy": 0.0,
                    "mean_latency_regret_ms": 251.0,
                    "task_count": 2,
                },
                max_loo_accuracy_drop=0.0,
                max_loo_latency_regret_increase_ms=0.0,
            )

        self.assertEqual(result["decision"], "quarantine")
        self.assertTrue(
            any("operational reference" in reason for reason in result["reasons"])
        )

    def test_preserve_accuracy_profile_quarantines_any_accuracy_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = self.write_dataset(tmpdir, "baseline.jsonl", ["qwen", "kimi"])
            candidate = self.write_dataset(tmpdir, "candidate.jsonl", ["qwen", "kimi", "glm"])
            thresholds = thresholds_for_profile(
                profile="preserve_accuracy",
                baseline_report=report(0.8, tasks=2, latency_regret_ms=500.0),
                min_loo_accuracy=0.7,
                max_loo_accuracy_drop=0.1,
                min_loo_solvable_pass_at_1=None,
                max_loo_latency_regret_ms=None,
                max_loo_latency_regret_increase_ms=None,
                min_loo_pass_at_1_vs_strongest=None,
            )

            result = evaluate_refresh(
                candidate_report=report(0.79, tasks=3, latency_regret_ms=400.0),
                candidate_dataset=candidate,
                baseline_report=report(0.8, tasks=2, latency_regret_ms=500.0),
                baseline_dataset=baseline,
                min_loo_accuracy=thresholds["min_loo_accuracy"],
                max_loo_accuracy_drop=thresholds["max_loo_accuracy_drop"],
                min_loo_solvable_pass_at_1=thresholds["min_loo_solvable_pass_at_1"],
                max_loo_latency_regret_ms=thresholds["max_loo_latency_regret_ms"],
                max_loo_latency_regret_increase_ms=thresholds["max_loo_latency_regret_increase_ms"],
                min_loo_pass_at_1_vs_strongest=thresholds["min_loo_pass_at_1_vs_strongest"],
            )

        self.assertEqual(result["decision"], "quarantine")
        self.assertTrue(any("below minimum" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
