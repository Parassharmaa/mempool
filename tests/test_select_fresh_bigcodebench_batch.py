import unittest

from tools.select_fresh_bigcodebench_batch import select_fresh_batch


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "prompt": prompt,
        "tests": [],
    }


class SelectFreshBigCodeBenchBatchTest(unittest.TestCase):
    def test_selects_preferred_categories_and_excludes_existing(self) -> None:
        tasks = [
            task("old", "Copy a file. You may use these Python libraries if helpful: ['shutil', 'os']."),
            task("fs", "Move a file. You may use these Python libraries if helpful: ['pathlib', 'os']."),
            task(
                "plot",
                "Draw a plot. You may use these Python libraries if helpful: ['matplotlib', 'collections'].",
            ),
            task(
                "data",
                "Group a dataframe. You may use these Python libraries if helpful: ['pandas', 'numpy'].",
            ),
        ]

        selection = select_fresh_batch(
            tasks,
            limit=3,
            exclude_ids={"old"},
            preferred_categories=("filesystem", "plotting", "datasci"),
        )

        self.assertEqual(selection["selected_task_ids"], ["fs", "plot", "data"])
        self.assertEqual(selection["candidate_count"], 3)

    def test_fills_remaining_slots_by_novelty(self) -> None:
        tasks = [
            task("a", "Copy a file. You may use these Python libraries if helpful: ['os']."),
            task("b", "Archive a file. You may use these Python libraries if helpful: ['zipfile']."),
            task("c", "Open SSL socket. You may use these Python libraries if helpful: ['ssl', 'socket']."),
        ]

        selection = select_fresh_batch(
            tasks,
            limit=3,
            preferred_categories=("filesystem",),
        )

        self.assertEqual(len(selection["selected_task_ids"]), 3)
        self.assertIn("c", selection["selected_task_ids"])

    def test_hard_strategy_prefers_risky_complex_tasks(self) -> None:
        tasks = [
            task("easy", "Count words. You may use these Python libraries if helpful: ['collections']."),
            task(
                "hard",
                "Fetch an archive over HTTP, unpack it, and plot results. "
                "You may use these Python libraries if helpful: ['requests', 'zipfile', 'matplotlib'].",
            ),
            task(
                "medium",
                "Group a dataframe. You may use these Python libraries if helpful: ['pandas', 'numpy'].",
            ),
        ]

        selection = select_fresh_batch(
            tasks,
            limit=1,
            strategy="hard",
        )

        self.assertEqual(selection["selected_task_ids"], ["hard"])


if __name__ == "__main__":
    unittest.main()
