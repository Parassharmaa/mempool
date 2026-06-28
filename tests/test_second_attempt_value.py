import unittest

from mempool.second_attempt_value import (
    evaluate_learned_value_head,
    evaluate_value_gated_fallback,
    leave_one_out_value_head_evaluation,
    second_attempt_value,
    train_second_attempt_value_head,
)


def record(second_passes: bool = True) -> dict:
    return {
        "task_id": "t1",
        "target": {"target_worker_id": "b"},
        "workers": [
            {"worker_id": "a", "pass_rate": 0.0, "mean_latency_ms": 100.0},
            {"worker_id": "b", "pass_rate": 1.0 if second_passes else 0.0, "mean_latency_ms": 500.0},
        ],
    }


class SecondAttemptValueTest(unittest.TestCase):
    def test_second_attempt_value_rewards_solve_gain_minus_latency_cost(self) -> None:
        prediction = {"worker_distribution": {"a": 0.6, "b": 0.4}}

        value = second_attempt_value(record(), prediction, latency_cost_per_second=0.1)

        self.assertEqual(value["solve_gain"], 1.0)
        self.assertAlmostEqual(value["latency_cost"], 0.05)
        self.assertAlmostEqual(value["value"], 0.95)
        self.assertEqual(value["label"], 1.0)

    def test_value_gated_fallback_skips_negative_value_second_attempt(self) -> None:
        prediction = {"worker_distribution": {"a": 0.6, "b": 0.4}}

        result = evaluate_value_gated_fallback(
            [record(second_passes=False)],
            [prediction],
            latency_cost_per_second=0.1,
        )

        self.assertEqual(result["fallbacks_taken"], 0)
        self.assertEqual(result["solved"], 0)

    def test_learned_value_head_can_trigger_fallback(self) -> None:
        prediction = {
            "worker_distribution": {"a": 0.6, "b": 0.4},
            "verifier_probability": 0.8,
            "abstain_probability": 0.0,
        }
        records = [record()]
        predictions = [prediction]

        head, history = train_second_attempt_value_head(
            records,
            predictions,
            latency_cost_per_second=0.1,
            epochs=30,
            learning_rate=0.1,
            threshold=0.1,
        )
        result = evaluate_learned_value_head(
            records,
            predictions,
            head,
            latency_cost_per_second=0.1,
        )

        self.assertTrue(history)
        self.assertEqual(result["fallbacks_taken"], 1)
        self.assertEqual(result["solved"], 1)

    def test_leave_one_out_value_head_reports_heldout_folds(self) -> None:
        records = [
            {
                "task_id": "positive",
                "target": {"target_worker_id": "b"},
                "workers": [
                    {"worker_id": "a", "pass_rate": 0.0, "mean_latency_ms": 100.0},
                    {"worker_id": "b", "pass_rate": 1.0, "mean_latency_ms": 500.0},
                ],
            },
            {
                "task_id": "negative",
                "target": {"target_worker_id": "a"},
                "workers": [
                    {"worker_id": "a", "pass_rate": 0.0, "mean_latency_ms": 100.0},
                    {"worker_id": "b", "pass_rate": 0.0, "mean_latency_ms": 500.0},
                ],
            },
            {
                "task_id": "easy",
                "target": {"target_worker_id": "a"},
                "workers": [
                    {"worker_id": "a", "pass_rate": 1.0, "mean_latency_ms": 100.0},
                    {"worker_id": "b", "pass_rate": 1.0, "mean_latency_ms": 500.0},
                ],
            },
        ]
        predictions = [
            {"worker_distribution": {"a": 0.6, "b": 0.4}, "verifier_probability": 0.8},
            {"worker_distribution": {"a": 0.6, "b": 0.4}, "verifier_probability": 0.1},
            {"worker_distribution": {"a": 0.6, "b": 0.4}, "verifier_probability": 0.2},
        ]

        result = leave_one_out_value_head_evaluation(
            records,
            predictions,
            latency_cost_per_second=0.1,
            thresholds=[0.1, 0.5],
            epochs=10,
            learning_rate=0.1,
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["policy"], "learned-second-attempt-value-head-loo")
        self.assertEqual(result["task_count"], 3)
        self.assertEqual(len(result["folds"]), 3)
        self.assertEqual(len(result["examples"]), 3)
        self.assertIn("selected_threshold", result["examples"][0])


if __name__ == "__main__":
    unittest.main()
