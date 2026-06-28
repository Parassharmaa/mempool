import unittest

from tools.select_canonical_specialist_batch import (
    select_canonical_specialist_batch,
    specialist_rank,
)


class StaticRouter:
    def distribution(self, record: dict) -> dict[str, float]:
        prompt = record["prompt"].lower()
        if "specialist-top" in prompt:
            return {"glm": 0.46, "qwen": 0.34, "deepseek": 0.12, "kimi": 0.08}
        if "archive" in prompt:
            return {"qwen": 0.43, "deepseek": 0.39, "glm": 0.10, "kimi": 0.08}
        if "glm" in prompt:
            return {"qwen": 0.44, "glm": 0.36, "deepseek": 0.12, "kimi": 0.08}
        return {"qwen": 0.70, "kimi": 0.20, "deepseek": 0.06, "glm": 0.04}


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "function_name": "task_func",
        "tests": [],
    }


class SelectCanonicalSpecialistBatchTest(unittest.TestCase):
    def test_specialist_rank_returns_first_specialist_position(self) -> None:
        self.assertEqual(
            specialist_rank(
                {"qwen": 0.5, "deepseek": 0.3, "glm": 0.2},
                {"deepseek", "glm"},
            ),
            2,
        )

    def test_prefers_fresh_filesystem_archive_with_specialist_pressure(self) -> None:
        tasks = [
            task(
                "seen",
                "Unpack an archive. You may use these Python libraries if helpful: ['zipfile'].",
            ),
            task(
                "archive",
                "Unpack an archive and copy files. "
                "You may use these Python libraries if helpful: ['zipfile', 'shutil', 'os'].",
            ),
            task(
                "generic",
                "Count words in a string. "
                "You may use these Python libraries if helpful: ['collections'].",
            ),
            task(
                "glm-file",
                "Read a file for specialist-top comparison. "
                "You may use these Python libraries if helpful: ['pathlib'].",
            ),
        ]

        selection = select_canonical_specialist_batch(
            tasks,
            StaticRouter(),
            limit=2,
            exclude_ids={"seen"},
            specialist_workers={"deepseek", "glm"},
        )

        self.assertEqual(selection["candidate_count"], 3)
        self.assertEqual(selection["selected_task_ids"][0], "archive")
        self.assertIn("glm-file", selection["selected_task_ids"])
        self.assertNotIn("seen", selection["selected_task_ids"])

    def test_can_require_specialist_top_rank(self) -> None:
        tasks = [
            task(
                "archive",
                "Unpack an archive and copy files. "
                "You may use these Python libraries if helpful: ['zipfile', 'shutil', 'os'].",
            ),
            task(
                "glm-file",
                "Read a file for specialist-top comparison. "
                "You may use these Python libraries if helpful: ['pathlib'].",
            ),
        ]

        selection = select_canonical_specialist_batch(
            tasks,
            StaticRouter(),
            limit=2,
            specialist_workers={"glm"},
            max_specialist_rank=1,
        )

        self.assertEqual(selection["max_specialist_rank"], 1)
        self.assertEqual(selection["selected_task_ids"], ["glm-file"])


if __name__ == "__main__":
    unittest.main()
