import unittest

from mempool.latency_safe_probe import (
    evaluate_probe_policy,
    probe_passed,
    sweep_probe_policies,
)


def record(task_id: str, pass_rates: dict[str, float]) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "unit",
        "prompt": "write a helper",
        "target_worker_id": max(pass_rates, key=pass_rates.get),
        "target_distribution": {worker_id: 1.0 / len(pass_rates) for worker_id in pass_rates},
        "workers": [
            {
                "worker_id": worker_id,
                "model": worker_id,
                "passed": pass_rate > 0.0,
                "pass_rate": pass_rate,
                "latency_ms": 100.0,
                "score": pass_rate,
                "reward": pass_rate,
                "cost_usd": 0.0,
                "failure_mode": None,
                "target_probability": 1.0 / len(pass_rates),
            }
            for worker_id, pass_rate in pass_rates.items()
        ],
    }


class LatencySafeProbeTest(unittest.TestCase):
    def test_single_probe_reports_confusion_metrics(self) -> None:
        records = [
            record("safe", {"probe": 1.0, "other": 1.0}),
            record("unsafe-visible", {"probe": 0.0, "other": 1.0}),
            record("unsafe-hidden", {"probe": 1.0, "other": 0.0}),
        ]

        result = evaluate_probe_policy(records, ["probe"])

        self.assertTrue(probe_passed(records[0], "probe"))
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)

    def test_pair_probe_all_mode_is_more_selective(self) -> None:
        records = [
            record("safe", {"a": 1.0, "b": 1.0, "c": 1.0}),
            record("unsafe-a", {"a": 1.0, "b": 0.0, "c": 1.0}),
            record("unsafe-b", {"a": 0.0, "b": 1.0, "c": 1.0}),
        ]

        result = evaluate_probe_policy(records, ["a", "b"], mode="all")
        swept = sweep_probe_policies(records, max_probe_count=2)

        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["tn"], 2)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(swept[0]["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
