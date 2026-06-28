import unittest

from mempool.multi_head_orchestrator import (
    MultiHeadOrchestrator,
    evaluate_multi_head_orchestrator,
    evaluate_multi_head_fallback_predictions,
    latency_regret_vector,
    leave_one_out_multi_head_evaluation,
    target_balance_weights,
    train_multi_head_orchestrator,
)


def example(task_id: str, keyword: str, target_worker: str) -> dict:
    worker_distribution = {"a": 0.95 if target_worker == "a" else 0.05, "b": 0.95 if target_worker == "b" else 0.05}
    return {
        "task_id": task_id,
        "target": {
            "worker_distribution": worker_distribution,
            "target_worker_id": target_worker,
            "workflow_distribution": {"direct": 1.0, "verify_then_fallback": 0.0},
            "workflow_kind": "direct",
            "verifier_probability": 0.1,
            "abstain_probability": 0.0,
        },
        "dense_features": {"bias": 1.0, keyword: 1.0},
        "workers": [
            {
                "worker_id": "a",
                "pass_rate": 1.0,
                "mean_latency_ms": 10.0 if target_worker == "a" else 50.0,
            },
            {
                "worker_id": "b",
                "pass_rate": 1.0,
                "mean_latency_ms": 10.0 if target_worker == "b" else 50.0,
            },
        ],
    }


class MultiHeadOrchestratorTest(unittest.TestCase):
    def test_training_learns_worker_head(self) -> None:
        records = [
            example("t1", "kw_file", "a"),
            example("t2", "kw_url", "b"),
        ]

        model, history = train_multi_head_orchestrator(
            records,
            epochs=120,
            learning_rate=0.05,
        )
        evaluation = evaluate_multi_head_orchestrator(records, model)

        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(evaluation["target_accuracy"], 1.0)
        self.assertEqual(evaluation["workflow_accuracy"], 1.0)

    def test_model_round_trips(self) -> None:
        records = [example("t1", "kw_file", "a")]
        model, _ = train_multi_head_orchestrator(records, epochs=5)

        restored = MultiHeadOrchestrator.from_dict(model.to_dict())

        self.assertEqual(restored.predict(records[0])["target_worker_id"], model.predict(records[0])["target_worker_id"])

    def test_leave_one_out_reports_metrics(self) -> None:
        records = [
            example("t1", "kw_file", "a"),
            example("t2", "kw_url", "b"),
            example("t3", "kw_url", "b"),
        ]

        result = leave_one_out_multi_head_evaluation(
            records,
            epochs=20,
            learning_rate=0.01,
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["task_count"], 3)
        self.assertIn("target_accuracy", result)

    def test_target_balance_weights_raise_rare_targets(self) -> None:
        records = [
            example("t1", "kw_file", "a"),
            example("t2", "kw_url", "a"),
            example("t3", "kw_archive", "b"),
        ]

        weights = target_balance_weights(records, power=1.0)

        self.assertGreater(weights["t3"], weights["t1"])
        self.assertAlmostEqual(sum(weights.values()) / len(weights), 1.0)

    def test_latency_regret_vector_is_relative_to_target_worker(self) -> None:
        record = example("t1", "kw_file", "a")

        regrets = latency_regret_vector(record, ["a", "b"])

        self.assertEqual(regrets[0], 0.0)
        self.assertGreater(regrets[1], 0.0)

    def test_latency_regret_weight_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            train_multi_head_orchestrator(
                [example("t1", "kw_file", "a")],
                latency_regret_weight=-1.0,
            )

    def test_multi_head_fallback_uses_verifier_and_margin_gates(self) -> None:
        record = example("t1", "kw_file", "b")
        record["workers"][0]["pass_rate"] = 0.0
        prediction = {
            "worker_distribution": {"a": 0.52, "b": 0.48},
            "verifier_probability": 0.8,
        }

        result = evaluate_multi_head_fallback_predictions(
            [record],
            [prediction],
            max_attempts=2,
            max_first_second_margin=0.1,
            min_verifier_probability=0.5,
        )

        self.assertEqual(result["solved"], 1)
        self.assertEqual(result["fallbacks_taken"], 1)
        self.assertEqual(result["examples"][0]["final_worker_id"], "b")

    def test_multi_head_fallback_blocks_low_verifier_probability(self) -> None:
        record = example("t1", "kw_file", "b")
        record["workers"][0]["pass_rate"] = 0.0
        prediction = {
            "worker_distribution": {"a": 0.52, "b": 0.48},
            "verifier_probability": 0.2,
        }

        result = evaluate_multi_head_fallback_predictions(
            [record],
            [prediction],
            max_attempts=2,
            max_first_second_margin=0.1,
            min_verifier_probability=0.5,
        )

        self.assertEqual(result["solved"], 0)
        self.assertEqual(result["fallbacks_taken"], 0)
        self.assertEqual(result["verifier_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
