import unittest
import tempfile
from pathlib import Path

from tools.select_similar_tasks import rank_similar_tasks, read_outcome_task_ids, task_similarity


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "prompt": prompt,
        "tests": [],
    }


class SelectSimilarTasksTest(unittest.TestCase):
    def test_similarity_prefers_library_overlap(self) -> None:
        seed = {
            "libraries": ["shutil", "random", "os"],
            "categories": ["filesystem"],
            "primary_category": "filesystem",
            "environment_risk": 0,
            "plausibility_score": 2.0,
        }
        close = {
            "libraries": ["shutil", "pathlib", "os"],
            "categories": ["filesystem"],
            "primary_category": "filesystem",
            "environment_risk": 0,
            "plausibility_score": 2.5,
        }
        far = {
            "libraries": ["urllib", "ssl"],
            "categories": ["network"],
            "primary_category": "network",
            "environment_risk": 3,
            "plausibility_score": 8.0,
        }

        self.assertGreater(task_similarity(seed, close), task_similarity(seed, far))

    def test_rank_similar_tasks_excludes_seed_and_existing(self) -> None:
        tasks = [
            task(
                "seed",
                "Move a random file. You may use these Python libraries if helpful: ['shutil', 'random', 'os'].",
            ),
            task(
                "close",
                "Copy files. You may use these Python libraries if helpful: ['shutil', 'pathlib', 'os'].",
            ),
            task(
                "excluded",
                "Move files. You may use these Python libraries if helpful: ['shutil', 'random', 'os'].",
            ),
            task(
                "far",
                "Open SSL socket. You may use these Python libraries if helpful: ['ssl', 'socket'].",
            ),
        ]

        ranked = rank_similar_tasks(tasks, "seed", exclude_ids={"excluded"})

        self.assertEqual(ranked[0]["task"]["id"], "close")
        self.assertNotIn("seed", [item["task"]["id"] for item in ranked])
        self.assertNotIn("excluded", [item["task"]["id"] for item in ranked])

    def test_reads_outcome_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "outcomes.jsonl"
            path.write_text(
                '{"task_id":"t1"}\n{"task_id":"t2"}\n{"task_id":"t1"}\n',
                encoding="utf-8",
            )

            task_ids = read_outcome_task_ids([path])

        self.assertEqual(task_ids, {"t1", "t2"})


if __name__ == "__main__":
    unittest.main()
