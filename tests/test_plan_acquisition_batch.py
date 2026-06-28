import unittest
from pathlib import Path

from tools.plan_acquisition_batch import (
    build_batch_manifest,
    select_batch,
    task_metadata_from_manifest,
    task_ids_for_target_worker,
    task_ids_from_summaries,
)


class PlanAcquisitionBatchTest(unittest.TestCase):
    def test_collects_screened_and_universal_failure_task_ids(self) -> None:
        screened, universal = task_ids_from_summaries(
            [
                {
                    "by_task": [
                        {"task_id": "t1", "universal_failure": True, "worker_count": 2},
                        {"task_id": "t2", "universal_failure": False, "worker_count": 2},
                    ],
                    "candidate_task_ids": ["t3"],
                }
            ],
            universal_min_workers=2,
        )

        self.assertEqual(screened, {"t1", "t2", "t3"})
        self.assertEqual(universal, {"t1"})

    def test_one_worker_screen_does_not_count_as_global_universal_failure(self) -> None:
        screened, universal = task_ids_from_summaries(
            [
                {
                    "by_task": [
                        {"task_id": "t1", "universal_failure": True, "worker_count": 1},
                    ],
                    "universal_failure_task_ids": ["t1"],
                }
            ],
            universal_min_workers=4,
        )

        self.assertEqual(screened, {"t1"})
        self.assertEqual(universal, set())

    def test_select_batch_skips_screened_tasks_in_source_order(self) -> None:
        tasks = [
            {"id": "t1"},
            {"id": "t2"},
            {"id": "t3"},
            {"id": "t4"},
        ]

        selected = select_batch(
            tasks,
            screened_task_ids={"t1"},
            universal_failure_task_ids={"t3"},
            batch_size=2,
        )

        self.assertEqual(selected, [{"id": "t2"}, {"id": "t4"}])

    def test_select_batch_can_filter_by_environment_risk(self) -> None:
        tasks = [
            {"id": "high"},
            {"id": "low"},
            {"id": "zero"},
        ]

        selected = select_batch(
            tasks,
            screened_task_ids=set(),
            universal_failure_task_ids=set(),
            batch_size=2,
            task_metadata={
                "high": {"environment_risk": 3},
                "low": {"environment_risk": 1},
                "zero": {"environment_risk": 0},
            },
            max_environment_risk=1,
        )

        self.assertEqual(selected, [{"id": "low"}, {"id": "zero"}])

    def test_select_batch_can_limit_to_allowed_task_ids(self) -> None:
        tasks = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        selected = select_batch(
            tasks,
            screened_task_ids=set(),
            universal_failure_task_ids=set(),
            batch_size=2,
            allowed_task_ids={"b", "c"},
        )

        self.assertEqual(selected, [{"id": "b"}, {"id": "c"}])

    def test_task_metadata_from_manifest_collects_selected_worker_items(self) -> None:
        metadata = task_metadata_from_manifest(
            {
                "selected_by_worker": {
                    "w1": [
                        {"task_id": "t1", "environment_risk": 3},
                    ],
                    "w2": [
                        {"task_id": "t2", "environment_risk": 0},
                    ],
                }
            }
        )

        self.assertEqual(metadata["t1"]["environment_risk"], 3)
        self.assertEqual(metadata["t2"]["environment_risk"], 0)

    def test_task_ids_for_target_worker_uses_manifest_groups(self) -> None:
        task_ids = task_ids_for_target_worker(
            {
                "selected_by_worker": {
                    "glm": [
                        {"task_id": "t1"},
                        {"task_id": "t2"},
                    ],
                    "kimi": [
                        {"task_id": "t3"},
                    ],
                }
            },
            "glm",
        )

        self.assertEqual(task_ids, {"t1", "t2"})

    def test_build_batch_manifest_includes_run_and_summary_commands(self) -> None:
        manifest = build_batch_manifest(
            source_manifest={
                "benchmark_id": "source",
                "repeat_count": 2,
                "worker_ids_to_run": ["w1", "w2"],
                "manifest_path": "source.json",
            },
            selected_tasks=[{"id": "t1"}, {"id": "t2"}],
            task_metadata={"t1": {"environment_risk": 1}},
            batch_tasks_output=Path("tasks.json"),
            run_id="run-1",
            output_path=Path("result.json"),
            outcomes_path=Path("result.jsonl"),
            worker_pool_path=Path("pool.json"),
            eval_timeout_seconds=20,
        )

        self.assertEqual(manifest["call_count"], 8)
        self.assertEqual(manifest["selected_task_ids"], ["t1", "t2"])
        self.assertEqual(manifest["selected_task_metadata"], {"t1": {"environment_risk": 1}})
        self.assertIn("--tasks tasks.json", manifest["run_command"])
        self.assertIn("result-summary.json", manifest["summary_command"])

    def test_build_batch_manifest_can_override_workers_and_repeat_count(self) -> None:
        manifest = build_batch_manifest(
            source_manifest={
                "benchmark_id": "source",
                "repeat_count": 2,
                "worker_ids_to_run": ["w1", "w2"],
            },
            selected_tasks=[{"id": "t1"}, {"id": "t2"}],
            task_metadata={},
            batch_tasks_output=Path("tasks.json"),
            run_id="screen",
            output_path=Path("screen.json"),
            outcomes_path=Path("screen.jsonl"),
            worker_pool_path=Path("qwen.json"),
            worker_ids_to_run=["qwen"],
            repeat_count=1,
            eval_timeout_seconds=20,
            benchmark_id="screen-benchmark",
        )

        self.assertEqual(manifest["benchmark_id"], "screen-benchmark")
        self.assertEqual(manifest["worker_ids_to_run"], ["qwen"])
        self.assertEqual(manifest["repeat_count"], 1)
        self.assertEqual(manifest["call_count"], 2)


if __name__ == "__main__":
    unittest.main()
