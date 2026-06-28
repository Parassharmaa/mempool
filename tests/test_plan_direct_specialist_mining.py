import unittest
from pathlib import Path

from tools.plan_direct_specialist_mining import (
    build_run_manifest,
    select_direct_mining_batch,
)


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "function_name": "task_func",
        "tests": [],
    }


class PlanDirectSpecialistMiningTest(unittest.TestCase):
    def test_selects_low_risk_fresh_tasks(self) -> None:
        tasks = [
            task("seen", "Use pathlib to copy a file"),
            task("low", "Use pathlib and csv to read files"),
            task("high", "Fetch a website. You may use these Python libraries if helpful: ['requests']."),
        ]

        selection = select_direct_mining_batch(
            tasks,
            exclude_task_ids={"seen"},
            limit=2,
            max_environment_risk=1,
            preferred_categories=["filesystem"],
        )

        self.assertEqual(selection["selected_task_ids"][0], "low")
        self.assertNotIn("seen", selection["selected_task_ids"])
        self.assertEqual(selection["candidate_count"], 2)

    def test_build_run_manifest_counts_worker_calls(self) -> None:
        manifest = build_run_manifest(
            selection={"selected_task_ids": ["a", "b"]},
            worker_pool={"workers": [{"id": "worker"}]},
            worker_pool_path=Path("pool.json"),
            tasks_output=Path("tasks.json"),
            run_id="run",
            output_path=Path("out.json"),
            outcomes_path=Path("out.jsonl"),
            repeat_count=1,
            eval_timeout_seconds=20,
        )

        self.assertEqual(manifest["call_count"], 2)
        self.assertEqual(manifest["worker_ids_to_run"], ["worker"])
        self.assertIn("--config pool.json", manifest["run_command"])


if __name__ == "__main__":
    unittest.main()
