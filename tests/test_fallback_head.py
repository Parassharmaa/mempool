import unittest

from mempool.fallback_head import evaluate_fallback_head, fallback_label, train_fallback_head
from mempool.logits_router import LogitsRouter


def record(task_id: str, second_passes: bool) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "b",
        "task_family": "bigcodebench_hard",
        "prompt": f"solve {task_id}",
        "prompt_features": {"categories": [], "libraries": [], "missing_libraries": []},
        "target_worker_id": "solver" if second_passes else "fast",
        "target_distribution": {"fast": 0.2, "solver": 0.8},
        "workers": [
            {
                "worker_id": "fast",
                "model": "fast",
                "passed": False,
                "score": 0.0,
                "latency_ms": 1,
                "cost_usd": 0.0,
                "failure_mode": "test_failure",
                "reward": -0.05,
                "target_probability": 0.2,
            },
            {
                "worker_id": "solver",
                "model": "solver",
                "passed": second_passes,
                "score": 1.0 if second_passes else 0.0,
                "latency_ms": 10,
                "cost_usd": 0.0,
                "failure_mode": None if second_passes else "test_failure",
                "reward": 0.95 if second_passes else -0.05,
                "target_probability": 0.8,
            },
        ],
    }


class FallbackHeadTest(unittest.TestCase):
    def test_fallback_label_marks_second_ranked_rescue(self) -> None:
        router = LogitsRouter(
            worker_ids=["fast", "solver"],
            feature_names=["bias"],
            weights=[[1.0], [0.9]],
        )

        self.assertEqual(fallback_label(record("yes", True), router), 1.0)
        self.assertEqual(fallback_label(record("no", False), router), 0.0)

    def test_trained_head_evaluates_second_attempts(self) -> None:
        router = LogitsRouter(
            worker_ids=["fast", "solver"],
            feature_names=["bias"],
            weights=[[1.0], [0.9]],
        )
        records = [record("yes", True), record("no", False)]

        head, history = train_fallback_head(
            records,
            router,
            epochs=50,
            learning_rate=0.05,
            threshold=0.0,
        )
        result = evaluate_fallback_head(records, router, head)

        self.assertTrue(history)
        self.assertEqual(result["fallback_opportunities"], 2)
        self.assertEqual(result["fallbacks_taken"], 2)
        self.assertEqual(result["useful_fallbacks"], 1)
        self.assertEqual(result["solvable_pass_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
