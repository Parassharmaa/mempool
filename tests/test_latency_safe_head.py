import unittest

from mempool.latency_safe_head import (
    evaluate_latency_safe_head,
    latency_safe_features,
    latency_safe_label,
    leave_one_out_latency_safe_evaluation,
    train_latency_safe_head,
    worker_reliability_context,
)
from mempool.logits_router import LogitsRouter


def record(task_id: str, prompt: str, pass_rates: dict[str, float]) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "unit",
        "prompt": prompt,
        "prompt_features": {"categories": ["filesystem"], "libraries": [], "missing_libraries": []},
        "target_worker_id": max(pass_rates, key=pass_rates.get),
        "target_distribution": {worker_id: 1.0 / len(pass_rates) for worker_id in pass_rates},
        "workers": [
            {
                "worker_id": worker_id,
                "model": worker_id,
                "passed": pass_rate > 0.0,
                "pass_rate": pass_rate,
                "score": pass_rate,
                "latency_ms": 100,
                "cost_usd": 0.0,
                "failure_mode": None,
                "reward": pass_rate,
                "target_probability": 1.0 / len(pass_rates),
            }
            for worker_id, pass_rate in pass_rates.items()
        ],
    }


class LatencySafeHeadTest(unittest.TestCase):
    def test_labels_broad_pass_rows_as_latency_safe(self) -> None:
        safe = record("safe", "zip files", {"a": 1.0, "b": 1.0})
        unsafe = record("unsafe", "network request", {"a": 1.0, "b": 0.0})

        self.assertEqual(latency_safe_label(safe), 1.0)
        self.assertEqual(latency_safe_label(unsafe), 0.0)

    def test_training_and_evaluation_report_metrics(self) -> None:
        records = [
            record("safe-1", "zip files", {"a": 1.0, "b": 1.0}),
            record("safe-2", "archive files", {"a": 1.0, "b": 1.0}),
            record("unsafe-1", "http request", {"a": 1.0, "b": 0.0}),
            record("unsafe-2", "network request", {"a": 0.0, "b": 1.0}),
        ]
        router = LogitsRouter(
            worker_ids=["a", "b"],
            feature_names=["bias"],
            weights=[[0.0], [0.0]],
        )

        head, history = train_latency_safe_head(records, router=router, epochs=20)
        evaluation = evaluate_latency_safe_head(records, head, router=router)
        loo = leave_one_out_latency_safe_evaluation(records, router=router, epochs=5)

        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(evaluation["task_count"], 4)
        self.assertEqual(evaluation["positive_count"], 2)
        self.assertTrue(loo["available"])
        self.assertEqual(loo["task_count"], 4)

    def test_reliability_features_are_available_without_label_leakage(self) -> None:
        records = [
            record("safe", "zip files", {"a": 1.0, "b": 1.0}),
            record("unsafe", "network request", {"a": 1.0, "b": 0.0}),
        ]
        router = LogitsRouter(
            worker_ids=["a", "b"],
            feature_names=["bias"],
            weights=[[0.0], [0.0]],
        )

        context = worker_reliability_context(records[:1])
        features = latency_safe_features(records[1], router=router, reliability_context=context)
        head, _ = train_latency_safe_head(
            records,
            router=router,
            epochs=5,
            use_reliability_features=True,
        )
        evaluation = evaluate_latency_safe_head(records, head, router=router, reliability_context=context)
        loo = leave_one_out_latency_safe_evaluation(
            records,
            router=router,
            epochs=5,
            use_reliability_features=True,
        )

        self.assertIn("reliability_mean_pass_rate", features)
        self.assertEqual(evaluation["task_count"], 2)
        self.assertTrue(loo["available"])


if __name__ == "__main__":
    unittest.main()
