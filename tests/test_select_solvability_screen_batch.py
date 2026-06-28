import unittest

from tools.select_solvability_screen_batch import select_screen_batch


def task(task_id: str) -> dict:
    return {"id": task_id, "prompt": f"solve {task_id}", "tests": []}


class SelectSolvabilityScreenBatchTest(unittest.TestCase):
    def test_selects_ranked_candidates_after_exclusions(self) -> None:
        acquisition_report = {
            "ranked_candidates": [
                {"task_id": "a", "score": 3.0},
                {"task_id": "b", "score": 2.0},
                {"task_id": "c", "score": 1.0},
            ]
        }
        selection = select_screen_batch(
            acquisition_report,
            [task("a"), task("b"), task("c")],
            limit=2,
            exclude_ids={"a"},
        )

        self.assertEqual(selection["candidate_count"], 2)
        self.assertEqual(selection["selected_task_ids"], ["b", "c"])
        self.assertEqual([item["id"] for item in selection["selected_tasks"]], ["b", "c"])


if __name__ == "__main__":
    unittest.main()
