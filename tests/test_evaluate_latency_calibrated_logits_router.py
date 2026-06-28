import unittest

from mempool.logits_router import LogitsRouter
from tools.evaluate_latency_calibrated_logits_router import (
    conditionally_calibrated_predictions,
    evaluate_grid,
    probe_gated_calibrate_task_ids,
)


def record() -> dict:
    return {
        "task_id": "t1",
        "benchmark_id": "bench",
        "task_family": "unit",
        "prompt": "zip files",
        "prompt_features": {},
        "target_worker_id": "fast",
        "target_distribution": {"slow": 0.52, "fast": 0.48},
        "workers": [
            {
                "worker_id": "slow",
                "model": "slow",
                "passed": True,
                "score": 1.0,
                "latency_ms": 10000,
                "cost_usd": 0.0,
                "failure_mode": None,
                "reward": 0.52,
                "target_probability": 0.52,
            },
            {
                "worker_id": "fast",
                "model": "fast",
                "passed": True,
                "score": 1.0,
                "latency_ms": 1000,
                "cost_usd": 0.0,
                "failure_mode": None,
                "reward": 0.48,
                "target_probability": 0.48,
            },
        ],
    }


class EvaluateLatencyCalibratedLogitsRouterTest(unittest.TestCase):
    def test_grid_can_select_latency_calibrated_candidate(self) -> None:
        router = LogitsRouter(
            worker_ids=["slow", "fast"],
            feature_names=["bias"],
            weights=[[0.52], [0.48]],
        )

        selected, candidates = evaluate_grid(
            records=[record()],
            router=router,
            latency_costs=[0.0, 0.01],
            min_probability_ratios=[0.0],
            min_probabilities=[0.0],
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(selected["pass_at_1"], 1.0)
        self.assertEqual(selected["mean_latency_regret_ms"], 0.0)

    def test_conditional_predictions_only_calibrate_listed_tasks(self) -> None:
        router = LogitsRouter(
            worker_ids=["slow", "fast"],
            feature_names=["bias"],
            weights=[[0.52], [0.48]],
        )
        latency_record = record()
        latency_record["task_id"] = "latency"
        risky_record = record()
        risky_record["task_id"] = "risky"
        risky_record["workers"][1]["passed"] = False
        risky_record["workers"][1]["score"] = 0.0

        predictions = conditionally_calibrated_predictions(
            records=[latency_record, risky_record],
            router=router,
            calibrate_task_ids={"latency"},
            latency_cost_per_second=0.01,
            min_probability_ratio=0.0,
            min_probability=0.0,
        )

        self.assertEqual(predictions[0]["worker_distribution"]["fast"], 1.0)
        self.assertEqual(predictions[1]["worker_distribution"]["slow"], 1.0)

    def test_probe_gate_derives_calibration_task_ids(self) -> None:
        safe_record = record()
        safe_record["task_id"] = "safe"
        safe_record["workers"][0]["worker_id"] = "probe"
        risky_record = record()
        risky_record["task_id"] = "risky"
        risky_record["workers"][0]["worker_id"] = "probe"
        risky_record["workers"][0]["passed"] = False
        risky_record["workers"][0]["pass_rate"] = 0.0

        task_ids = probe_gated_calibrate_task_ids(
            records=[safe_record, risky_record],
            probe_worker_ids=["probe"],
        )

        self.assertEqual(task_ids, {"safe"})


if __name__ == "__main__":
    unittest.main()
