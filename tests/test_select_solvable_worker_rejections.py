import unittest

from tools.select_solvable_worker_rejections import (
    max_similarity,
    seed_analyses_from_outcomes,
    select_solvable_worker_rejections,
)


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "function_name": "task_func",
        "tests": [],
    }


class SelectSolvableWorkerRejectionsTest(unittest.TestCase):
    def test_builds_positive_and_universal_fail_seeds_from_specialist_rows(self) -> None:
        rows = [
            {
                "task_id": "positive",
                "worker_id": "specialist",
                "passed": True,
                "prompt": "Read a csv file and write json.",
            },
            {
                "task_id": "failed",
                "worker_id": "specialist",
                "passed": False,
                "prompt": "Use windows subprocess path constants.",
            },
            {
                "task_id": "ignored",
                "worker_id": "qwen",
                "passed": True,
                "prompt": "Read csv.",
            },
        ]

        positive, universal_fail = seed_analyses_from_outcomes(
            rows,
            specialist_workers={"specialist"},
        )

        self.assertEqual(len(positive), 1)
        self.assertEqual(len(universal_fail), 1)
        self.assertIn("filesystem", positive[0]["categories"])

    def test_selects_qwen_rejections_near_positive_and_away_from_all_fail(self) -> None:
        tasks = [
            task("csv-json", "Read a csv file and write a json file."),
            task("windows-subprocess", "Use windows subprocess path constants and execute a backup."),
            task("qwen-fast", "Read a csv file and count rows."),
        ]
        qwen_rows = [
            {"task_id": "csv-json", "worker_id": "qwen", "passed": False, "latency_ms": 1000},
            {"task_id": "windows-subprocess", "worker_id": "qwen", "passed": False, "latency_ms": 1000},
            {"task_id": "qwen-fast", "worker_id": "qwen", "passed": True, "latency_ms": 1000},
        ]
        prior_rows = [
            {
                "task_id": "positive",
                "worker_id": "specialist",
                "passed": True,
                "prompt": "Read a csv file and write a json file.",
            },
            {
                "task_id": "failed",
                "worker_id": "specialist",
                "passed": False,
                "prompt": "Use windows subprocess path constants and execute a backup.",
            },
        ]

        selection = select_solvable_worker_rejections(
            tasks=tasks,
            qwen_rows=qwen_rows,
            prior_rows=prior_rows,
            rejected_worker_id="qwen",
            specialist_workers={"specialist"},
            limit=2,
            min_gate_score=-1.0,
        )

        self.assertEqual(selection["selected_task_ids"], ["csv-json"])
        self.assertEqual(selection["rejected_task_count"], 2)
        self.assertEqual(selection["positive_seed_count"], 1)
        self.assertEqual(selection["universal_fail_seed_count"], 1)
        self.assertEqual(len(selection["scored_rejections"]), 2)

    def test_negative_similarity_is_neutral_not_a_bonus(self) -> None:
        seed = {
            "libraries": ["subprocess"],
            "categories": ["subprocess"],
            "primary_category": "subprocess",
            "environment_risk": 3,
            "plausibility_score": 10,
        }
        unrelated = {
            "libraries": ["csv"],
            "categories": ["filesystem"],
            "primary_category": "filesystem",
            "environment_risk": 0,
            "plausibility_score": 1,
        }

        self.assertEqual(max_similarity([seed], unrelated), 0.0)

    def test_can_cap_universal_failure_similarity(self) -> None:
        tasks = [
            task("mixed", "Read a json file with random values and update collections."),
        ]
        qwen_rows = [
            {"task_id": "mixed", "worker_id": "qwen", "passed": False, "latency_ms": 1000},
        ]
        prior_rows = [
            {
                "task_id": "positive",
                "worker_id": "specialist",
                "passed": True,
                "prompt": "Read a json file with random values and update collections.",
            },
            {
                "task_id": "failed",
                "worker_id": "specialist",
                "passed": False,
                "prompt": "Read a json file with random values and update collections.",
            },
        ]

        uncapped = select_solvable_worker_rejections(
            tasks=tasks,
            qwen_rows=qwen_rows,
            prior_rows=prior_rows,
            rejected_worker_id="qwen",
            specialist_workers={"specialist"},
            limit=1,
        )
        capped = select_solvable_worker_rejections(
            tasks=tasks,
            qwen_rows=qwen_rows,
            prior_rows=prior_rows,
            rejected_worker_id="qwen",
            specialist_workers={"specialist"},
            limit=1,
            max_universal_fail_similarity=0.1,
        )

        self.assertEqual(uncapped["selected_task_ids"], ["mixed"])
        self.assertEqual(capped["selected_task_ids"], [])
        self.assertEqual(capped["rejected_by_universal_fail_cap"], 1)


if __name__ == "__main__":
    unittest.main()
