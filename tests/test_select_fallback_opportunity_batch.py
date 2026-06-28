import unittest

from mempool.logits_router import LogitsRouter
from tools.select_fallback_opportunity_batch import select_fallback_opportunity_batch


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "tests": [],
    }


class SelectFallbackOpportunityBatchTest(unittest.TestCase):
    def test_prefers_low_margin_candidate_and_excludes_seen_tasks(self) -> None:
        router = LogitsRouter(
            worker_ids=["qwen", "glm"],
            feature_names=["bias", "kw_file"],
            weights=[[0.0, 0.0], [-0.2, 0.1]],
        )
        tasks = [
            task("seen", "Copy a file. You may use these Python libraries if helpful: ['os']."),
            task("low", "Archive a file. You may use these Python libraries if helpful: ['zipfile']."),
            task("high", "Count words. You may use these Python libraries if helpful: ['collections']."),
        ]

        selection = select_fallback_opportunity_batch(
            tasks,
            router,
            limit=2,
            exclude_ids={"seen"},
        )

        self.assertEqual(selection["candidate_count"], 2)
        self.assertEqual(selection["selected_task_ids"][0], "low")
        self.assertNotIn("seen", selection["selected_task_ids"])

    def test_seed_similarity_can_rank_related_candidate_first(self) -> None:
        router = LogitsRouter(
            worker_ids=["qwen", "deepseek"],
            feature_names=["bias"],
            weights=[[0.0], [0.0]],
        )
        tasks = [
            task(
                "similar",
                "Unpack a zip archive and move files. "
                "You may use these Python libraries if helpful: ['zipfile', 'shutil', 'os'].",
            ),
            task(
                "different",
                "Draw a plot from a dataframe. "
                "You may use these Python libraries if helpful: ['pandas', 'matplotlib'].",
            ),
        ]
        seed = {
            "libraries": ["zipfile", "shutil", "os"],
            "categories": ["filesystem"],
            "primary_category": "filesystem",
            "environment_risk": 0,
            "plausibility_score": 2.0,
        }

        selection = select_fallback_opportunity_batch(
            tasks,
            router,
            limit=1,
            seeds=[seed],
            preferred_second_workers={"deepseek"},
        )

        self.assertEqual(selection["selected_task_ids"], ["similar"])


if __name__ == "__main__":
    unittest.main()
