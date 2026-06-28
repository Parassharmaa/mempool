import unittest

from tools.plan_solvability_aware_specialist_acquisition import read_screened_task_ids


class PlanSolvabilityAwareSpecialistAcquisitionTest(unittest.TestCase):
    def test_read_screened_task_ids_collects_task_level_summary_ids(self) -> None:
        # Covered through the pure helper by patching a tiny fake Path-like object.
        class FakePath:
            def read_text(self, encoding: str) -> str:
                return (
                    '{"by_task":[{"task_id":"a"}],'
                    '"records":[{"task_id":"d"}],'
                    '"universal_failure_task_ids":["b"],'
                    '"candidate_task_ids":["c"]}'
                )

        self.assertEqual(read_screened_task_ids(FakePath()), {"a", "b", "c", "d"})

    def test_read_screened_task_ids_accepts_legacy_by_task_mapping(self) -> None:
        class FakePath:
            def read_text(self, encoding: str) -> str:
                return '{"by_task":{"a":{"passed":0,"total":4},"b":{"passed":1,"total":4}}}'

        self.assertEqual(read_screened_task_ids(FakePath()), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
