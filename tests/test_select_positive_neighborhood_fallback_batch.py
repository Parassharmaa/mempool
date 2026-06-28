import unittest

from mempool.logits_router import LogitsRouter
from tools.select_positive_neighborhood_fallback_batch import (
    select_positive_neighborhood_batch,
)


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "tests": [],
    }


class SelectPositiveNeighborhoodFallbackBatchTest(unittest.TestCase):
    def test_prefers_candidate_similar_to_positive_seed(self) -> None:
        router = LogitsRouter(
            worker_ids=["qwen", "kimi"],
            feature_names=["bias"],
            weights=[[0.0], [0.0]],
        )
        positive_seed = {
            "libraries": ["pandas", "glob", "os"],
            "categories": ["filesystem", "datasci"],
            "primary_category": "filesystem",
            "environment_risk": 0,
            "plausibility_score": 2.0,
        }
        tasks = [
            task(
                "similar",
                "Concatenate CSV files into a dataframe. "
                "You may use these Python libraries if helpful: ['pandas', 'glob', 'os'].",
            ),
            task(
                "far",
                "Open an SSL socket and send email. "
                "You may use these Python libraries if helpful: ['ssl', 'socket', 'smtplib'].",
            ),
        ]

        selection = select_positive_neighborhood_batch(
            tasks,
            router,
            [positive_seed],
            limit=1,
            preferred_second_workers={"kimi"},
        )

        self.assertEqual(selection["selected_task_ids"], ["similar"])
        self.assertGreater(selection["selected"][0]["positive_similarity"], 0.0)

    def test_excludes_seen_tasks(self) -> None:
        router = LogitsRouter(
            worker_ids=["qwen", "kimi"],
            feature_names=["bias"],
            weights=[[0.0], [0.0]],
        )
        positive_seed = {
            "libraries": ["os"],
            "categories": ["filesystem"],
            "primary_category": "filesystem",
            "environment_risk": 0,
            "plausibility_score": 1.0,
        }
        tasks = [
            task("seen", "Copy a file. You may use these Python libraries if helpful: ['os']."),
            task("fresh", "Move a file. You may use these Python libraries if helpful: ['os']."),
        ]

        selection = select_positive_neighborhood_batch(
            tasks,
            router,
            [positive_seed],
            limit=2,
            exclude_ids={"seen"},
        )

        self.assertEqual(selection["selected_task_ids"], ["fresh"])


if __name__ == "__main__":
    unittest.main()
