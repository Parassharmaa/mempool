import unittest

from tools.select_active_rescue_acquisition_batch import select_active_rescue_acquisition_batch


class StaticRouter:
    def distribution(self, record: dict) -> dict[str, float]:
        prompt = record["prompt"].lower()
        if "deep rescue" in prompt:
            return {"qwen": 0.48, "deepseek": 0.44, "kimi": 0.08}
        if "hard trap" in prompt:
            return {"qwen": 0.50, "kimi": 0.42, "deepseek": 0.08}
        return {"qwen": 0.62, "kimi": 0.24, "deepseek": 0.14}


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "function_name": "solve",
        "prompt": prompt,
        "tests": [],
    }


def corpus_record(
    task_id: str,
    prompt: str,
    *,
    useful: bool,
    top: str = "qwen",
    second: str = "deepseek",
    alternate: str | None = "deepseek",
) -> dict:
    return {
        "task_id": task_id,
        "prompt": prompt,
        "prompt_features": {
            "categories": ["filesystem"],
            "libraries": ["pathlib"],
            "missing_libraries": [],
        },
        "top_worker_id": top,
        "second_worker_id": second,
        "best_ranked_alternate_worker_id": alternate if useful else None,
        "fastest_passed_alternate_worker_id": alternate if useful else None,
        "useful_any_fallback": useful,
        "useful_second_fallback": useful and alternate == second,
        "hard_negative": not useful,
    }


class SelectActiveRescueAcquisitionBatchTest(unittest.TestCase):
    def test_prefers_same_top_alternate_rescue_over_hard_negative_neighbor(self) -> None:
        tasks = [
            task(
                "candidate-rescue",
                "Use pathlib for deep rescue file walking and archive metadata.",
            ),
            task(
                "candidate-hard",
                "Use pathlib for hard trap file walking and archive metadata.",
            ),
            task("seen-positive", "Use pathlib for deep rescue file walking."),
        ]
        corpus = [
            corpus_record(
                "seen-positive",
                "Use pathlib for deep rescue file walking.",
                useful=True,
                top="qwen",
                second="deepseek",
                alternate="deepseek",
            ),
            corpus_record(
                "hard-negative",
                "Use pathlib for hard trap file walking.",
                useful=False,
                top="qwen",
                second="kimi",
                alternate=None,
            ),
        ]

        selection = select_active_rescue_acquisition_batch(
            tasks=tasks,
            router=StaticRouter(),
            corpus_records=corpus,
            exclude_ids=set(),
            limit=1,
        )

        self.assertEqual(selection["selected_task_ids"], ["candidate-rescue"])
        selected = selection["selected"][0]
        self.assertGreater(selected["positive_pair_similarity"], 0.0)
        self.assertEqual(selected["top_worker"], "qwen")
        self.assertEqual(selected["second_worker"], "deepseek")
        self.assertNotIn("seen-positive", selection["selected_task_ids"])
        self.assertEqual(selection["excluded_count"], 2)


if __name__ == "__main__":
    unittest.main()
