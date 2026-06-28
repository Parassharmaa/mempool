import unittest

from mempool.conditional_policy import evaluate_conditional_fallback, evaluate_gated_fallback
from mempool.logits_router import LogitsRouter


class ConditionalPolicyTest(unittest.TestCase):
    def test_fallback_attempts_second_ranked_worker_after_failure(self) -> None:
        record = {
            "task_id": "t1",
            "task_family": "bigcodebench_hard",
            "prompt": "solve",
            "prompt_features": {"categories": [], "libraries": [], "missing_libraries": []},
            "target_worker_id": "solver",
            "target_distribution": {"fast": 0.4, "solver": 0.6},
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
                    "target_probability": 0.4,
                },
                {
                    "worker_id": "solver",
                    "model": "solver",
                    "passed": True,
                    "score": 1.0,
                    "latency_ms": 10,
                    "cost_usd": 0.0,
                    "failure_mode": None,
                    "reward": 0.95,
                    "target_probability": 0.6,
                },
            ],
        }
        router = LogitsRouter(
            worker_ids=["fast", "solver"],
            feature_names=["bias"],
            weights=[[1.0], [0.0]],
        )

        result = evaluate_conditional_fallback([record], router, max_attempts=2)

        self.assertEqual(result["solved"], 1)
        self.assertEqual(result["solvable_pass_at_1"], 1.0)
        self.assertEqual(result["mean_attempts"], 2.0)
        self.assertEqual(
            [attempt["worker_id"] for attempt in result["examples"][0]["attempts"]],
            ["fast", "solver"],
        )

    def test_gated_fallback_skips_second_attempt_when_margin_is_large(self) -> None:
        record = {
            "task_id": "t1",
            "task_family": "bigcodebench_hard",
            "prompt": "solve",
            "prompt_features": {"categories": [], "libraries": [], "missing_libraries": []},
            "target_worker_id": "solver",
            "target_distribution": {"fast": 0.4, "solver": 0.6},
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
                    "target_probability": 0.4,
                },
                {
                    "worker_id": "solver",
                    "model": "solver",
                    "passed": True,
                    "score": 1.0,
                    "latency_ms": 10,
                    "cost_usd": 0.0,
                    "failure_mode": None,
                    "reward": 0.95,
                    "target_probability": 0.6,
                },
            ],
        }
        router = LogitsRouter(
            worker_ids=["fast", "solver"],
            feature_names=["bias"],
            weights=[[2.0], [0.0]],
        )

        result = evaluate_gated_fallback(
            [record],
            router,
            max_attempts=2,
            max_first_second_margin=0.1,
        )

        self.assertEqual(result["solved"], 0)
        self.assertEqual(result["fallbacks_taken"], 0)


if __name__ == "__main__":
    unittest.main()
