import tempfile
import unittest
from pathlib import Path

from tools.select_router_miss_neighborhood_batch import (
    select_router_miss_neighborhood_batch,
)


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "function_name": "task_func",
        "tests": [],
    }


class SelectRouterMissNeighborhoodBatchTest(unittest.TestCase):
    def test_selects_round_robin_from_miss_neighborhoods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "routing.jsonl"
            dataset.write_text(
                "\n".join(
                    [
                        '{"task_id":"seed-zip","prompt":"Zip files with pathlib and os.","workers":[],"target_distribution":{},"target_worker_id":"glm"}',
                        '{"task_id":"seed-random","prompt":"Create random integers and compute statistics mean.","workers":[],"target_distribution":{},"target_worker_id":"qwen"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            miss_plan = {
                "dataset": str(dataset),
                "neighborhoods": [
                    {
                        "task_ids": ["seed-zip"],
                        "target_worker_id": "glm",
                        "predicted_worker_id": "qwen",
                        "category_key": "filesystem",
                        "top_library_keys": [{"libraries": "pathlib+zipfile+os", "count": 1}],
                    },
                    {
                        "task_ids": ["seed-random"],
                        "target_worker_id": "qwen",
                        "predicted_worker_id": "kimi",
                        "category_key": "general",
                        "top_library_keys": [{"libraries": "random+statistics", "count": 1}],
                    },
                ],
            }
            tasks = [
                task("seed-zip", "Zip files with pathlib and os."),
                task("zip-close", "Archive files with pathlib zipfile os."),
                task("random-close", "Use random values and statistics mean."),
                task("far", "Open an HTTP URL with requests."),
            ]

            selection = select_router_miss_neighborhood_batch(
                tasks=tasks,
                miss_plan=miss_plan,
                exclude_ids=set(),
                limit=2,
                per_neighborhood_limit=1,
            )

        self.assertEqual(selection["selected_task_ids"], ["zip-close", "random-close"])
        self.assertEqual(selection["selected"][0]["target_worker_id"], "glm")
        self.assertEqual(selection["selected"][1]["target_worker_id"], "qwen")

    def test_excludes_seen_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "routing.jsonl"
            dataset.write_text(
                '{"task_id":"seed","prompt":"Move files with shutil os.","workers":[],"target_distribution":{},"target_worker_id":"deepseek"}\n',
                encoding="utf-8",
            )
            miss_plan = {
                "dataset": str(dataset),
                "neighborhoods": [
                    {
                        "task_ids": ["seed"],
                        "target_worker_id": "deepseek",
                        "predicted_worker_id": "kimi",
                        "category_key": "filesystem",
                        "top_library_keys": [{"libraries": "shutil+os", "count": 1}],
                    }
                ],
            }
            tasks = [
                task("seed", "Move files with shutil os."),
                task("excluded", "Move files with shutil pathlib os."),
                task("kept", "Copy files with shutil pathlib os."),
            ]

            selection = select_router_miss_neighborhood_batch(
                tasks=tasks,
                miss_plan=miss_plan,
                exclude_ids={"excluded"},
                limit=1,
            )

        self.assertEqual(selection["selected_task_ids"], ["kept"])

    def test_can_fallback_to_seed_repeat_batch_when_no_fresh_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "routing.jsonl"
            dataset.write_text(
                '{"task_id":"seed","prompt":"Zip files with pathlib os.","workers":[],"target_distribution":{},"target_worker_id":"glm"}\n',
                encoding="utf-8",
            )
            miss_plan = {
                "dataset": str(dataset),
                "neighborhoods": [
                    {
                        "task_ids": ["seed"],
                        "target_worker_id": "glm",
                        "predicted_worker_id": "qwen",
                        "category_key": "filesystem",
                        "top_library_keys": [{"libraries": "pathlib+os", "count": 1}],
                    }
                ],
            }
            tasks = [task("seed", "Zip files with pathlib os.")]

            selection = select_router_miss_neighborhood_batch(
                tasks=tasks,
                miss_plan=miss_plan,
                exclude_ids={"seed"},
                limit=1,
                fallback_to_seed_tasks=True,
            )

        self.assertTrue(selection["fallback_used"])
        self.assertEqual(selection["selected_task_ids"], ["seed"])
        self.assertEqual(selection["selected"][0]["selection_reason"], "fallback_seed_repeat")


if __name__ == "__main__":
    unittest.main()
