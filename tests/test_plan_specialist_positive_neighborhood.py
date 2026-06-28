import unittest

from tools.plan_specialist_positive_neighborhood import (
    select_specialist_positive_neighborhood,
    worker_positive_seed_analyses,
)


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "function_name": "task_func",
        "tests": [],
    }


class PlanSpecialistPositiveNeighborhoodTest(unittest.TestCase):
    def test_worker_positive_seeds_only_include_target_worker_passes(self) -> None:
        records = [
            {
                "task_id": "seed",
                "prompt": "Use pathlib to read files",
                "workers": [
                    {"worker_id": "glm", "passed": True},
                    {"worker_id": "kimi", "passed": False, "pass_rate": 0.0},
                ],
            }
        ]

        self.assertEqual(len(worker_positive_seed_analyses(records, "glm")), 1)
        self.assertEqual(worker_positive_seed_analyses(records, "kimi"), [])

    def test_selects_fresh_tasks_near_worker_positive_rows(self) -> None:
        records = [
            {
                "task_id": "seed",
                "prompt": "Use pandas and pathlib to merge CSV files",
                "workers": [{"worker_id": "glm", "passed": True, "pass_rate": 1.0}],
            }
        ]
        tasks = [
            task("seed", "Use pandas and pathlib to merge CSV files"),
            task("similar", "Use pandas and pathlib to clean CSV files"),
            task("far", "Open an ssl socket and send an email"),
        ]

        selection = select_specialist_positive_neighborhood(
            tasks=tasks,
            routing_records=records,
            target_workers=["glm"],
            exclude_task_ids=set(),
            per_worker_limit=1,
        )

        self.assertEqual(selection["seed_counts_by_worker"], {"glm": 1})
        self.assertEqual(selection["selected_task_ids"], ["similar"])
        self.assertGreater(
            selection["selected_by_worker"]["glm"][0]["positive_similarity"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
