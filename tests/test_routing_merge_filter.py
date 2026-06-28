import unittest

from mempool.routing_merge_filter import filter_merge_ready_records


def record(task_id: str, target: str, pass_rates: dict[str, float]) -> dict:
    return {
        "task_id": task_id,
        "target_worker_id": target,
        "workers": [
            {
                "worker_id": worker_id,
                "pass_rate": pass_rate,
                "passed": pass_rate > 0,
            }
            for worker_id, pass_rate in pass_rates.items()
        ],
    }


class RoutingMergeFilterTest(unittest.TestCase):
    def test_keeps_only_stable_non_all_fail_targets(self) -> None:
        kept, report = filter_merge_ready_records(
            [
                record("stable", "a", {"a": 1.0, "b": 0.0}),
                record("unstable", "a", {"a": 0.5, "b": 0.0}),
                record("all-fail", "a", {"a": 0.0, "b": 0.0}),
            ]
        )

        self.assertEqual([item["task_id"] for item in kept], ["stable"])
        self.assertEqual(report["kept_records"], 1)
        self.assertEqual(report["dropped_records"], 2)


if __name__ == "__main__":
    unittest.main()
