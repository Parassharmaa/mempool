import unittest

from mempool.mined_fallback_head import (
    evaluate_head,
    leave_one_out_evaluation,
    select_threshold,
    train_mined_fallback_head,
)


def mined_record(task_id: str, useful: bool, margin: float = 0.01) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "bigcodebench_hard",
        "prompt": "write code with files" if useful else "write code with http",
        "prompt_features": {
            "categories": ["filesystem"] if useful else ["network"],
            "libraries": ["pathlib"] if useful else ["requests"],
            "missing_libraries": [],
        },
        "source_dataset": "dataset.jsonl",
        "top_worker_id": "qwen",
        "top_probability": 0.5 + margin,
        "top_passed": False,
        "top_latency_ms": 10,
        "second_worker_id": "deepseek" if useful else "kimi",
        "second_probability": 0.5,
        "second_passed": useful,
        "second_latency_ms": 20,
        "first_second_margin": margin,
        "fallback_opportunity": True,
        "useful_second_fallback": useful,
        "useful_any_fallback": useful,
        "hard_negative": not useful,
        "best_ranked_alternate_worker_id": "deepseek" if useful else None,
        "best_ranked_alternate_rank": 2 if useful else None,
        "best_ranked_alternate_probability": 0.5 if useful else None,
        "best_ranked_alternate_latency_ms": 20 if useful else None,
        "additional_latency_to_best_ranked_alternate_ms": 20 if useful else None,
        "total_latency_to_best_ranked_alternate_ms": 30 if useful else None,
        "target_worker_id": "deepseek" if useful else "qwen",
        "target_worker_passed": useful,
        "solvable_by_any_worker": useful,
        "alternate_count": 3,
        "passed_alternate_count": 1 if useful else 0,
        "alternates": [],
    }


class MinedFallbackHeadTest(unittest.TestCase):
    def test_trains_and_evaluates_mined_fallback_labels(self) -> None:
        records = [
            mined_record("positive-a", True),
            mined_record("positive-b", True),
            mined_record("negative-a", False),
            mined_record("negative-b", False),
        ]

        head, history = train_mined_fallback_head(
            records,
            epochs=120,
            learning_rate=0.05,
            l2=0.0,
        )
        probabilities = [head.probability(record) for record in records]
        threshold, metrics = select_threshold(records, probabilities, [0.2, 0.4, 0.5])
        head.threshold = threshold
        evaluation = evaluate_head(records, head)

        self.assertTrue(history)
        self.assertGreaterEqual(metrics["f1"], 0.8)
        self.assertGreaterEqual(evaluation["recall"], 0.5)
        self.assertEqual(evaluation["positive_count"], 2)

    def test_leave_one_out_reports_fold_metrics(self) -> None:
        records = [
            mined_record("positive-a", True),
            mined_record("positive-b", True),
            mined_record("negative-a", False),
            mined_record("negative-b", False),
        ]

        report = leave_one_out_evaluation(
            records,
            thresholds=[0.2, 0.4, 0.5],
            epochs=40,
            learning_rate=0.05,
            l2=0.0,
        )

        self.assertEqual(report["task_count"], 4)
        self.assertEqual(len(report["folds"]), 4)
        self.assertIn("f1", report["metrics"])


if __name__ == "__main__":
    unittest.main()
