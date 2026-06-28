import unittest

from mempool.latency_calibrated_router import (
    evaluate_latency_calibrated_predictions,
    latency_calibrated_worker_choice,
)


def record(target_worker: str = "fast") -> dict:
    return {
        "task_id": "t1",
        "target": {"target_worker_id": target_worker},
        "workers": [
            {"worker_id": "slow", "pass_rate": 1.0, "mean_latency_ms": 10000.0},
            {"worker_id": "fast", "pass_rate": 1.0, "mean_latency_ms": 1000.0},
            {"worker_id": "cheap_fail", "pass_rate": 0.0, "mean_latency_ms": 100.0},
        ],
    }


def routing_record(target_worker: str = "fast") -> dict:
    payload = record(target_worker=target_worker)
    payload["target_worker_id"] = payload.pop("target")["target_worker_id"]
    return payload


class LatencyCalibratedRouterTest(unittest.TestCase):
    def test_latency_cost_can_choose_slightly_lower_probability_fast_worker(self) -> None:
        prediction = {
            "worker_distribution": {
                "slow": 0.52,
                "fast": 0.48,
                "cheap_fail": 0.01,
            }
        }

        choice = latency_calibrated_worker_choice(
            record(),
            prediction,
            latency_cost_per_second=0.01,
            min_probability_ratio=0.8,
        )

        self.assertEqual(choice["selected_worker_id"], "fast")

    def test_probability_ratio_blocks_too_low_probability_cheap_worker(self) -> None:
        prediction = {
            "worker_distribution": {
                "slow": 0.8,
                "fast": 0.1,
                "cheap_fail": 0.1,
            }
        }

        choice = latency_calibrated_worker_choice(
            record(target_worker="slow"),
            prediction,
            latency_cost_per_second=0.1,
            min_probability_ratio=0.5,
        )

        self.assertEqual(choice["selected_worker_id"], "slow")

    def test_evaluation_reports_latency_regret_and_change_rate(self) -> None:
        prediction = {
            "worker_distribution": {
                "slow": 0.52,
                "fast": 0.48,
                "cheap_fail": 0.01,
            }
        }

        result = evaluate_latency_calibrated_predictions(
            [record()],
            [prediction],
            latency_cost_per_second=0.01,
            min_probability_ratio=0.8,
        )

        self.assertEqual(result["target_accuracy"], 1.0)
        self.assertEqual(result["pass_at_1"], 1.0)
        self.assertEqual(result["changed_from_top"], 1)
        self.assertEqual(result["mean_latency_regret_ms"], 0.0)

    def test_evaluation_accepts_routing_record_target_shape(self) -> None:
        prediction = {
            "worker_distribution": {
                "slow": 0.52,
                "fast": 0.48,
                "cheap_fail": 0.01,
            }
        }

        result = evaluate_latency_calibrated_predictions(
            [routing_record()],
            [prediction],
            latency_cost_per_second=0.01,
            min_probability_ratio=0.8,
        )

        self.assertEqual(result["target_accuracy"], 1.0)
        self.assertEqual(result["examples"][0]["target_worker_id"], "fast")

    def test_rejects_negative_latency_cost(self) -> None:
        with self.assertRaises(ValueError):
            latency_calibrated_worker_choice(
                record(),
                {"worker_distribution": {"slow": 1.0}},
                latency_cost_per_second=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
