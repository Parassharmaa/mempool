import unittest

from tools.select_fallback_error_neighborhood_batch import (
    loo_error_task_ids,
    select_error_neighborhood_batch,
)


class StaticRouter:
    def distribution(self, record: dict) -> dict[str, float]:
        prompt = record["prompt"].lower()
        if "glm" in prompt:
            return {"qwen": 0.45, "glm": 0.40, "deepseek": 0.15}
        if "deepseek" in prompt:
            return {"qwen": 0.44, "deepseek": 0.41, "glm": 0.15}
        return {"qwen": 0.60, "kimi": 0.30, "deepseek": 0.10}


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "prompt": prompt,
        "family": "bigcodebench_hard",
        "function_name": "task_func",
        "tests": [],
    }


def seed(seed_task_id: str, prompt: str, worker: str | None = None) -> dict:
    return {
        "task_id": seed_task_id,
        "seed_task_id": seed_task_id,
        "prompt_length": len(prompt),
        "test_length": 0,
        "libraries": ["pathlib"] if "file" in prompt else ["requests"],
        "missing_libraries": [],
        "categories": ["filesystem"] if "file" in prompt else ["network"],
        "primary_category": "filesystem" if "file" in prompt else "network",
        "environment_risk": 0 if "file" in prompt else 3,
        "plausibility_score": 1.0,
        "best_ranked_alternate_worker_id": worker,
    }


class SelectFallbackErrorNeighborhoodBatchTest(unittest.TestCase):
    def test_extracts_loo_error_ids(self) -> None:
        report = {
            "leave_one_out": {
                "metrics": {
                    "examples": [
                        {
                            "task_id": "fn",
                            "actual_useful_fallback": True,
                            "predicted_fallback": False,
                        },
                        {
                            "task_id": "fp",
                            "actual_useful_fallback": False,
                            "predicted_fallback": True,
                        },
                    ]
                }
            }
        }

        self.assertEqual(
            loo_error_task_ids(report),
            {"false_negative": {"fn"}, "false_positive": {"fp"}},
        )

    def test_selects_balanced_error_neighborhoods(self) -> None:
        tasks = [
            task("candidate-deepseek", "Use pathlib to inspect files for deepseek rescue."),
            task("candidate-glm", "Use pathlib to inspect files for glm rescue."),
            task("candidate-network", "Use requests to download network content."),
            task("candidate-http", "Use requests for http parsing."),
        ]
        false_negative_seeds = [
            seed("missed-368", "file archive task", "deepseek"),
            seed("missed-963", "file csv task", "glm"),
        ]
        false_positive_seeds = [seed("false-857", "network request task")]

        selection = select_error_neighborhood_batch(
            tasks,
            StaticRouter(),
            false_negative_seeds,
            false_positive_seeds,
            limit=4,
        )

        groups = {item["target_group"] for item in selection["selected"]}
        self.assertEqual(selection["candidate_count"], 4)
        self.assertIn("missed-positive-neighborhood", groups)
        self.assertIn("false-positive-neighborhood", groups)
        self.assertEqual(len(selection["selected_task_ids"]), 4)


if __name__ == "__main__":
    unittest.main()
