import unittest

from mempool.logits_router import LogitsRouter
from tools.mine_historical_fallback_cases import mine_record, summarize


def routing_record(
    task_id: str = "task",
    top_passed: bool = False,
    second_passed: bool = False,
    third_passed: bool = False,
) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "bigcodebench_hard",
        "prompt": "Write code",
        "prompt_features": {},
        "target_worker_id": "third" if third_passed else "top",
        "target_distribution": {"top": 0.7, "second": 0.2, "third": 0.1},
        "workers": [
            {
                "worker_id": "top",
                "model": "top",
                "passed": top_passed,
                "score": 1.0 if top_passed else 0.0,
                "latency_ms": 10,
                "cost_usd": 0.0,
                "failure_mode": None if top_passed else "test_failure",
                "reward": 1.0 if top_passed else 0.0,
                "target_probability": 0.7,
            },
            {
                "worker_id": "second",
                "model": "second",
                "passed": second_passed,
                "score": 1.0 if second_passed else 0.0,
                "latency_ms": 20,
                "cost_usd": 0.0,
                "failure_mode": None if second_passed else "test_failure",
                "reward": 1.0 if second_passed else 0.0,
                "target_probability": 0.2,
            },
            {
                "worker_id": "third",
                "model": "third",
                "passed": third_passed,
                "score": 1.0 if third_passed else 0.0,
                "latency_ms": 5,
                "cost_usd": 0.0,
                "failure_mode": None if third_passed else "test_failure",
                "reward": 1.0 if third_passed else 0.0,
                "target_probability": 0.1,
            },
        ],
    }


class MineHistoricalFallbackCasesTest(unittest.TestCase):
    def test_mines_second_fallback_positive(self) -> None:
        router = LogitsRouter(
            worker_ids=["top", "second", "third"],
            feature_names=["bias"],
            weights=[[2.0], [1.0], [0.0]],
        )

        mined = mine_record(routing_record(second_passed=True), router, "dataset.jsonl")

        self.assertIsNotNone(mined)
        assert mined is not None
        self.assertTrue(mined["fallback_opportunity"])
        self.assertTrue(mined["useful_second_fallback"])
        self.assertTrue(mined["useful_any_fallback"])
        self.assertFalse(mined["hard_negative"])
        self.assertEqual(mined["best_ranked_alternate_worker_id"], "second")
        self.assertEqual(mined["best_ranked_alternate_rank"], 2)
        self.assertEqual(mined["additional_latency_to_best_ranked_alternate_ms"], 20)
        self.assertEqual(mined["total_latency_to_best_ranked_alternate_ms"], 30)

    def test_mines_later_rank_positive_and_fastest_alternate(self) -> None:
        router = LogitsRouter(
            worker_ids=["top", "second", "third"],
            feature_names=["bias"],
            weights=[[2.0], [1.0], [0.0]],
        )

        mined = mine_record(routing_record(third_passed=True), router, "dataset.jsonl")

        self.assertIsNotNone(mined)
        assert mined is not None
        self.assertFalse(mined["useful_second_fallback"])
        self.assertTrue(mined["useful_any_fallback"])
        self.assertEqual(mined["best_ranked_alternate_worker_id"], "third")
        self.assertEqual(mined["best_ranked_alternate_rank"], 3)
        self.assertEqual(mined["fastest_passed_alternate_worker_id"], "third")

    def test_skips_top_passes_and_summarizes_hard_negatives(self) -> None:
        router = LogitsRouter(
            worker_ids=["top", "second", "third"],
            feature_names=["bias"],
            weights=[[2.0], [1.0], [0.0]],
        )

        skipped = mine_record(routing_record(top_passed=True), router, "dataset.jsonl")
        negative = mine_record(routing_record(task_id="hard"), router, "dataset.jsonl")

        self.assertIsNone(skipped)
        self.assertIsNotNone(negative)
        assert negative is not None
        report = summarize([negative], [])
        self.assertEqual(report["fallback_opportunities"], 1)
        self.assertEqual(report["useful_any_fallbacks"], 0)
        self.assertEqual(report["hard_negatives"], 1)
        self.assertEqual(report["hard_negative_task_ids"], ["hard"])

    def test_filters_unmeasured_router_workers(self) -> None:
        router = LogitsRouter(
            worker_ids=["missing", "top", "second"],
            feature_names=["bias"],
            weights=[[3.0], [2.0], [1.0]],
        )

        mined = mine_record(routing_record(second_passed=True), router, "dataset.jsonl")

        self.assertIsNotNone(mined)
        assert mined is not None
        self.assertEqual(mined["top_worker_id"], "top")
        self.assertEqual(mined["second_worker_id"], "second")
        self.assertTrue(mined["useful_second_fallback"])


if __name__ == "__main__":
    unittest.main()
