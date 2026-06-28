import unittest

from mempool.logits_router import LogitsRouter
from tools.build_fallback_training_dataset import build_fallback_records, summarize


def routing_record(second_passed: bool) -> dict:
    return {
        "task_id": "t1",
        "benchmark_id": "b",
        "task_family": "bigcodebench_hard",
        "prompt": "solve",
        "prompt_features": {},
        "target_worker_id": "second" if second_passed else "top",
        "target_distribution": {"top": 0.7, "second": 0.3},
        "workers": [
            {
                "worker_id": "top",
                "model": "top",
                "passed": False,
                "score": 0.0,
                "latency_ms": 10,
                "cost_usd": 0.0,
                "failure_mode": "test_failure",
                "reward": 0.0,
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
                "target_probability": 0.3,
            },
        ],
    }


class BuildFallbackTrainingDatasetTest(unittest.TestCase):
    def test_builds_useful_and_hurt_fallback_labels(self) -> None:
        router = LogitsRouter(
            worker_ids=["top", "second"],
            feature_names=["bias"],
            weights=[[1.0], [0.0]],
        )

        records = build_fallback_records(
            [routing_record(True), routing_record(False)],
            router,
        )
        report = summarize(records)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["fallback_label"], 1.0)
        self.assertTrue(records[0]["useful_fallback"])
        self.assertEqual(records[1]["fallback_label"], 0.0)
        self.assertTrue(records[1]["fallback_hurt"])
        self.assertEqual(report["fallback_opportunity_count"], 2)
        self.assertEqual(report["useful_fallback_count"], 1)
        self.assertEqual(report["fallback_hurt_count"], 1)


if __name__ == "__main__":
    unittest.main()
