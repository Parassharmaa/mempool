import unittest

from mempool.routing_merge_audit import audit_routing_merge_readiness


def record(task_id: str, target: str, workers: list[dict]) -> dict:
    return {
        "task_id": task_id,
        "target_worker_id": target,
        "workers": workers,
    }


class RoutingMergeAuditTest(unittest.TestCase):
    def test_accepts_stable_solvable_targets(self) -> None:
        report = audit_routing_merge_readiness(
            [
                record(
                    "task",
                    "strong",
                    [
                        {"worker_id": "strong", "pass_rate": 1.0},
                        {"worker_id": "weak", "pass_rate": 0.0},
                    ],
                )
            ]
        )

        self.assertTrue(report["ready_to_merge"])
        self.assertEqual(report["reasons"], [])

    def test_rejects_all_fail_and_unstable_targets(self) -> None:
        report = audit_routing_merge_readiness(
            [
                record(
                    "all-fail",
                    "fast",
                    [
                        {"worker_id": "fast", "pass_rate": 0.0},
                        {"worker_id": "slow", "pass_rate": 0.0},
                    ],
                ),
                record(
                    "partial",
                    "maybe",
                    [
                        {"worker_id": "maybe", "pass_rate": 0.5},
                        {"worker_id": "bad", "pass_rate": 0.0},
                    ],
                ),
            ]
        )

        self.assertFalse(report["ready_to_merge"])
        self.assertEqual(report["all_fail_tasks"], ["all-fail"])
        self.assertEqual(report["unstable_target_tasks"][0]["task_id"], "partial")

    def test_can_allow_all_fail_tasks_for_negative_dataset(self) -> None:
        report = audit_routing_merge_readiness(
            [
                record(
                    "all-fail",
                    "fast",
                    [{"worker_id": "fast", "pass_rate": 0.0}],
                )
            ],
            allow_all_fail_tasks=True,
        )

        self.assertTrue(report["ready_to_merge"])


if __name__ == "__main__":
    unittest.main()
