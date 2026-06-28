import unittest

from mempool.acquisition_plan import (
    build_catalog_candidate_acquisition_plan,
    build_solvability_aware_specialist_plan,
    build_specialist_acquisition_plan,
    equivalent_task_id,
    solvability_profile,
)


class AcquisitionPlanTest(unittest.TestCase):
    def test_selects_unmeasured_workers_and_requested_tasks(self) -> None:
        pool = {
            "base_url": "https://example.test/v1",
            "api_key_env": "KEY_ENV",
            "workers": [
                {"id": "measured", "model": "old"},
                {"id": "new", "model": "new-model"},
            ],
        }
        tasks = [
            {"id": "task-a", "prompt": "a", "family": "f", "function_name": "f", "tests": []},
            {"id": "task-b", "prompt": "b", "family": "f", "function_name": "f", "tests": []},
        ]

        plan = build_catalog_candidate_acquisition_plan(
            current_pool=pool,
            task_sources=tasks,
            task_ids=["task-b"],
            measured_worker_ids={"measured"},
            repeat_count=2,
        )

        self.assertEqual(plan["selected_task_ids"], ["task-b"])
        self.assertEqual(plan["selected_worker_ids"], ["new"])
        self.assertEqual(plan["call_count"], 2)
        self.assertEqual(plan["worker_pool"]["workers"][0]["model"], "new-model")

    def test_raises_for_missing_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "task not found"):
            build_catalog_candidate_acquisition_plan(
                current_pool={"workers": []},
                task_sources=[],
                task_ids=["missing"],
                measured_worker_ids=set(),
                repeat_count=1,
            )

    def test_equivalent_bigcodebench_task_ids(self) -> None:
        self.assertTrue(equivalent_task_id("BigCodeBench/123", "bigcodebench-hard-BigCodeBench-123"))

    def test_builds_specialist_plan_from_missed_predictions(self) -> None:
        tasks = [
            {
                "id": "BigCodeBench/10",
                "prompt": "Use pandas and requests to scrape a table",
                "family": "bigcodebench_hard",
                "function_name": "task_func",
                "tests": [],
            },
            {
                "id": "BigCodeBench/20",
                "prompt": "Use pandas and requests for another table",
                "family": "bigcodebench_hard",
                "function_name": "task_func",
                "tests": [],
            },
            {
                "id": "BigCodeBench/30",
                "prompt": "Simple string task",
                "family": "bigcodebench_hard",
                "function_name": "task_func",
                "tests": [],
            },
        ]
        routing_records = [
            {
                "task_id": "bigcodebench-hard-BigCodeBench-10",
                "target_worker_id": "deepseek",
                "workers": [{"worker_id": "deepseek"}],
            }
        ]
        report = {
            "leave_one_out": {
                "predictions": [
                    {
                        "task_id": "bigcodebench-hard-BigCodeBench-10",
                        "target_worker_id": "deepseek",
                        "predicted_worker_id": "qwen",
                    }
                ]
            }
        }

        plan = build_specialist_acquisition_plan(
            task_sources=tasks,
            routing_records=routing_records,
            candidate_report=report,
            target_workers=["deepseek"],
            comparison_workers=["qwen"],
            exclude_task_ids=set(),
            per_worker_limit=1,
            repeat_count=2,
        )

        self.assertEqual(plan["seed_miss_count"], 1)
        self.assertEqual(plan["selected_task_ids"], ["BigCodeBench/20"])
        self.assertEqual(plan["worker_ids_to_run"], ["deepseek", "qwen"])
        self.assertEqual(plan["call_count"], 4)

    def test_solvability_profile_counts_positive_records(self) -> None:
        profile = solvability_profile(
            [
                {
                    "prompt_features": {
                        "categories": ["filesystem"],
                        "libraries": ["pathlib"],
                    },
                    "workers": [{"passed": True}],
                },
                {
                    "prompt_features": {
                        "categories": ["network"],
                        "libraries": ["requests"],
                    },
                    "workers": [{"passed": False, "pass_rate": 0.0}],
                },
            ]
        )

        self.assertEqual(profile["positive_count"], 1)
        self.assertEqual(profile["category_counts"], {"filesystem": 1})
        self.assertEqual(profile["library_counts"], {"pathlib": 1})

    def test_solvability_aware_plan_prefers_historically_positive_shape(self) -> None:
        tasks = [
            {
                "id": "BigCodeBench/10",
                "prompt": "Use pathlib to read files from a directory",
                "family": "bigcodebench_hard",
                "function_name": "task_func",
                "tests": [],
            },
            {
                "id": "BigCodeBench/20",
                "prompt": "Use requests to download a remote zip archive",
                "family": "bigcodebench_hard",
                "function_name": "task_func",
                "tests": [],
            },
            {
                "id": "BigCodeBench/30",
                "prompt": "Use pathlib to inspect another directory",
                "family": "bigcodebench_hard",
                "function_name": "task_func",
                "tests": [],
            },
        ]
        routing_records = [
            {
                "task_id": "bigcodebench-hard-BigCodeBench-10",
                "prompt_features": {
                    "categories": ["filesystem"],
                    "libraries": ["pathlib"],
                    "environment_risk": 0,
                    "plausibility_score": 1.0,
                },
                "workers": [{"passed": True, "pass_rate": 1.0}],
            }
        ]
        report = {
            "leave_one_out": {
                "predictions": [
                    {
                        "task_id": "bigcodebench-hard-BigCodeBench-10",
                        "target_worker_id": "glm",
                        "predicted_worker_id": "qwen",
                    }
                ]
            }
        }

        plan = build_solvability_aware_specialist_plan(
            task_sources=tasks,
            routing_records=routing_records,
            candidate_report=report,
            target_workers=["glm"],
            exclude_task_ids=set(),
            per_worker_limit=1,
        )

        self.assertEqual(plan["positive_prior_count"], 1)
        self.assertEqual(plan["selected_task_ids"], ["BigCodeBench/30"])
        self.assertGreater(
            plan["selected_by_worker"]["glm"][0]["solvability_score"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
