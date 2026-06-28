import unittest

from tools.select_qwen_fail_contrast_batch import (
    score_candidate,
    select_qwen_fail_contrast_batch,
)


class StaticRouter:
    def distribution(self, record: dict) -> dict[str, float]:
        prompt = record["prompt"].lower()
        if "specialist top" in prompt:
            return {
                "ollama-cloud-kimi-k2.7-code": 0.42,
                "ollama-cloud-qwen3-coder-480b": 0.35,
                "ollama-cloud-glm-5.2": 0.13,
                "ollama-cloud-deepseek-v4-pro": 0.10,
            }
        return {
            "ollama-cloud-qwen3-coder-480b": 0.42,
            "ollama-cloud-kimi-k2.7-code": 0.35,
            "ollama-cloud-glm-5.2": 0.13,
            "ollama-cloud-deepseek-v4-pro": 0.10,
        }


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "function_name": "task_func",
        "tests": [],
    }


class SelectQwenFailContrastBatchTest(unittest.TestCase):
    def test_score_prefers_fail_seed_similarity_over_anchor_similarity(self) -> None:
        fail_seed = {
            "libraries": ["pandas", "statsmodels", "matplotlib"],
            "categories": ["filesystem", "plotting", "datasci"],
            "primary_category": "filesystem",
            "environment_risk": 3,
            "plausibility_score": 8.3,
        }
        anchor_seed = {
            "libraries": ["pandas", "sklearn", "matplotlib"],
            "categories": ["plotting", "datasci"],
            "primary_category": "plotting",
            "environment_risk": 3,
            "plausibility_score": 6.9,
        }
        routing = {
            "top_worker": "ollama-cloud-qwen3-coder-480b",
            "second_worker": "ollama-cloud-kimi-k2.7-code",
            "first_second_margin": 0.05,
            "distribution": {
                "ollama-cloud-qwen3-coder-480b": 0.36,
                "ollama-cloud-kimi-k2.7-code": 0.31,
                "ollama-cloud-glm-5.2": 0.2,
                "ollama-cloud-deepseek-v4-pro": 0.13,
            },
        }

        fail_like_score, fail_like_details = score_candidate(
            analysis=fail_seed,
            routing=routing,
            qwen_fail_seeds=[fail_seed],
            qwen_anchor_seeds=[anchor_seed],
            specialist_workers={"ollama-cloud-kimi-k2.7-code"},
            qwen_worker="ollama-cloud-qwen3-coder-480b",
        )
        anchor_like_score, anchor_like_details = score_candidate(
            analysis=anchor_seed,
            routing=routing,
            qwen_fail_seeds=[fail_seed],
            qwen_anchor_seeds=[anchor_seed],
            specialist_workers={"ollama-cloud-kimi-k2.7-code"},
            qwen_worker="ollama-cloud-qwen3-coder-480b",
        )

        self.assertGreater(fail_like_details["qwen_fail_similarity"], fail_like_details["qwen_anchor_similarity"])
        self.assertGreater(anchor_like_details["qwen_anchor_similarity"], anchor_like_details["qwen_fail_similarity"])
        self.assertGreater(fail_like_score, anchor_like_score)

    def test_score_rejects_empty_limit_in_selector(self) -> None:
        with self.assertRaises(ValueError):
            select_qwen_fail_contrast_batch(
                tasks=[],
                router=object(),
                exclude_ids=set(),
                qwen_fail_seeds=[],
                qwen_anchor_seeds=[],
                specialist_workers=set(),
                qwen_worker="qwen",
                limit=0,
            )

    def test_can_require_specialist_top_rank(self) -> None:
        selection = select_qwen_fail_contrast_batch(
            tasks=[
                task("qwen-top", "Use csv files."),
                task("specialist-top", "Use csv files with specialist top behavior."),
            ],
            router=StaticRouter(),
            exclude_ids=set(),
            qwen_fail_seeds=[
                {
                    "libraries": ["csv"],
                    "categories": ["filesystem"],
                    "primary_category": "filesystem",
                    "environment_risk": 0,
                    "plausibility_score": 1.0,
                }
            ],
            qwen_anchor_seeds=[],
            specialist_workers={"ollama-cloud-kimi-k2.7-code"},
            qwen_worker="ollama-cloud-qwen3-coder-480b",
            limit=4,
            max_specialist_rank=1,
        )

        self.assertEqual(selection["max_specialist_rank"], 1)
        self.assertEqual(selection["selected_task_ids"], ["specialist-top"])
        self.assertEqual(selection["selected"][0]["specialist_rank"], 1)


if __name__ == "__main__":
    unittest.main()
