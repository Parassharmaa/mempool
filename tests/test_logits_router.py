import unittest

from mempool.logits_router import (
    LogitsRouter,
    evaluate_logits_router,
    kl_divergence,
    leave_one_out_logits_evaluation,
    reward_target_vector,
    target_vector,
    train_logits_router,
    training_weight,
)


def record(task_id: str, prompt: str, target: dict[str, float]) -> dict:
    return {
        "task_id": task_id,
        "task_family": "bigcodebench_hard",
        "prompt": prompt,
        "prompt_features": {"categories": ["filesystem"], "libraries": [], "missing_libraries": []},
        "target_worker_id": max(target, key=target.get),
        "target_distribution": target,
        "workers": [
            {
                "worker_id": worker_id,
                "passed": True,
                "latency_ms": 10,
                "cost_usd": 0.0,
                "reward": target[worker_id],
            }
            for worker_id in sorted(target)
        ],
    }


class LogitsRouterTest(unittest.TestCase):
    def test_training_reduces_kl_on_soft_targets(self) -> None:
        records = [
            record("t1", "zip files from a directory", {"a": 0.8, "b": 0.2}),
            record("t2", "scan ftp network subprocess", {"a": 0.2, "b": 0.8}),
        ]
        initial_router = LogitsRouter.initialize(records)
        initial_kl = sum(
            kl_divergence(
                target_vector(item, initial_router.worker_ids),
                [
                    initial_router.distribution(item)[worker_id]
                    for worker_id in initial_router.worker_ids
                ],
            )
            for item in records
        )

        router, history = train_logits_router(records, epochs=100, learning_rate=0.0001)
        trained_kl = evaluate_logits_router(records, router)["mean_kl"] * len(records)

        self.assertLess(trained_kl, initial_kl)
        self.assertGreaterEqual(len(history), 2)

    def test_reward_target_vector_uses_reward_temperature(self) -> None:
        records = [
            record("t1", "zip files from a directory", {"a": 0.5, "b": 0.5}),
        ]
        records[0]["workers"][0]["reward"] = 0.0
        records[0]["workers"][1]["reward"] = 1.0

        target = reward_target_vector(records[0], ["a", "b"], temperature=0.25)

        self.assertLess(target[0], 0.1)
        self.assertGreater(target[1], 0.9)

    def test_training_can_use_reward_targets(self) -> None:
        records = [
            record("t1", "zip files from a directory", {"a": 0.5, "b": 0.5}),
            record("t2", "scan ftp network subprocess", {"a": 0.5, "b": 0.5}),
        ]
        records[0]["workers"][0]["reward"] = 1.0
        records[0]["workers"][1]["reward"] = 0.0
        records[1]["workers"][0]["reward"] = 0.0
        records[1]["workers"][1]["reward"] = 1.0

        router, _ = train_logits_router(
            records,
            epochs=50,
            learning_rate=0.01,
            target_mode="reward",
            reward_temperature=0.2,
        )

        self.assertEqual(router.predict(records[0]), "a")
        self.assertEqual(router.predict(records[1]), "b")

    def test_training_weight_downweights_conflicting_rows(self) -> None:
        records = [
            record("base-1", "zip files from a directory", {"a": 1.0, "b": 0.0}),
            record("base-2", "zip files from a directory", {"a": 1.0, "b": 0.0}),
            record("new-conflict", "zip files from a directory", {"a": 0.0, "b": 1.0}),
        ]
        records[2]["training_weight"] = 0.01

        self.assertEqual(training_weight(records[2]), 0.01)

        router, _ = train_logits_router(
            records,
            epochs=80,
            learning_rate=0.01,
            l2=0.0,
            target_mode="reward",
            reward_temperature=0.2,
        )

        self.assertEqual(router.predict(records[0]), "a")

    def test_model_round_trips(self) -> None:
        records = [record("t1", "zip files from a directory", {"a": 0.8, "b": 0.2})]
        router, _ = train_logits_router(records, epochs=5, learning_rate=0.0001)

        restored = LogitsRouter.from_dict(router.to_dict())

        self.assertEqual(restored.predict(records[0]), router.predict(records[0]))

    def test_initialize_can_warm_start_matching_weights(self) -> None:
        initial = LogitsRouter(
            worker_ids=["a", "b"],
            feature_names=["bias", "kw_zip"],
            weights=[[1.0, 2.0], [3.0, 4.0]],
        )
        records = [
            record("t1", "zip files from a directory", {"a": 0.8, "b": 0.2}),
            record("t2", "scan network", {"a": 0.2, "b": 0.8}),
        ]

        router = LogitsRouter.initialize(records, initial_router=initial)

        self.assertEqual(router.weights[router.worker_ids.index("a")][router.feature_names.index("bias")], 1.0)
        self.assertEqual(router.weights[router.worker_ids.index("b")][router.feature_names.index("kw_zip")], 4.0)
        self.assertEqual(router.weights[router.worker_ids.index("a")][router.feature_names.index("kw_scan")], 0.0)

    def test_leave_one_out_requires_multiple_records(self) -> None:
        records = [record("t1", "zip files from a directory", {"a": 0.8, "b": 0.2})]

        result = leave_one_out_logits_evaluation(records)

        self.assertFalse(result["available"])

    def test_leave_one_out_reports_examples(self) -> None:
        records = [
            record("t1", "zip files from a directory", {"a": 0.8, "b": 0.2}),
            record("t2", "scan ftp network subprocess", {"a": 0.2, "b": 0.8}),
        ]

        result = leave_one_out_logits_evaluation(records, epochs=10, learning_rate=0.01, l2=0.0)

        self.assertEqual(result["task_count"], 2)
        self.assertEqual(len(result["examples"]), 2)
        self.assertEqual(len(result["predictions"]), 2)
        self.assertIn("mean_latency_regret_ms", result)
        self.assertIn("solvable_pass_at_1", result)

    def test_evaluation_includes_latency_regret(self) -> None:
        records = [
            {
                "task_id": "t1",
                "task_family": "bigcodebench_hard",
                "prompt": "zip files from a directory",
                "prompt_features": {"categories": ["filesystem"], "libraries": [], "missing_libraries": []},
                "target_worker_id": "fast",
                "target_distribution": {"fast": 0.2, "slow": 0.8},
                "workers": [
                    {"worker_id": "fast", "passed": True, "latency_ms": 10, "cost_usd": 0.0},
                    {"worker_id": "slow", "passed": True, "latency_ms": 50, "cost_usd": 0.0},
                ],
            }
        ]
        router = LogitsRouter(
            worker_ids=["fast", "slow"],
            feature_names=["bias"],
            weights=[[0.0], [1.0]],
        )

        result = evaluate_logits_router(records, router)

        self.assertEqual(result["predictions"], ["slow"])
        self.assertEqual(result["mean_latency_ms"], 50.0)
        self.assertEqual(result["mean_target_latency_ms"], 10.0)
        self.assertEqual(result["mean_latency_regret_ms"], 40.0)
        self.assertEqual(result["solvable_task_count"], 1)
        self.assertEqual(result["solvable_pass_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
