import unittest

from mempool.logits_router import LogitsRouter
from tools.select_corpus_guided_fallback_batch import select_corpus_guided_fallback_batch


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "tests": [],
    }


def corpus_record(task_id: str, useful: bool, prompt: str) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bigcodebench-hard",
        "task_family": "bigcodebench_hard",
        "prompt": prompt,
        "prompt_features": {
            "categories": ["filesystem"] if useful else ["network"],
            "libraries": ["zipfile"] if useful else ["requests"],
            "missing_libraries": [],
        },
        "top_worker_id": "qwen",
        "second_worker_id": "deepseek" if useful else "kimi",
        "second_latency_ms": 1000,
        "useful_any_fallback": useful,
        "useful_second_fallback": useful,
        "hard_negative": not useful,
        "best_ranked_alternate_worker_id": "deepseek" if useful else None,
        "fastest_passed_alternate_worker_id": "deepseek" if useful else None,
    }


class CorpusGuidedFallbackBatchTest(unittest.TestCase):
    def test_prefers_positive_similarity_and_excludes_seen_corpus_tasks(self) -> None:
        router = LogitsRouter(
            worker_ids=["qwen", "deepseek"],
            feature_names=["bias"],
            weights=[[0.0], [0.0]],
        )
        tasks = [
            task(
                "seen-positive",
                "Unzip archive files. You may use these Python libraries if helpful: ['zipfile'].",
            ),
            task(
                "similar",
                "Extract a zip archive. You may use these Python libraries if helpful: ['zipfile'].",
            ),
            task(
                "different",
                "Download a web page. You may use these Python libraries if helpful: ['requests'].",
            ),
        ]
        corpus = [
            corpus_record(
                "seen-positive",
                True,
                "Unzip archive files. You may use these Python libraries if helpful: ['zipfile'].",
            ),
            corpus_record(
                "hard-negative",
                False,
                "Download a web page. You may use these Python libraries if helpful: ['requests'].",
            ),
        ]

        selection = select_corpus_guided_fallback_batch(
            tasks=tasks,
            router=router,
            corpus_records=corpus,
            exclude_ids=set(),
            limit=1,
        )

        self.assertEqual(selection["selected_task_ids"], ["similar"])
        self.assertNotIn("seen-positive", selection["selected_task_ids"])
        self.assertEqual(selection["positive_seed_count"], 1)
        self.assertEqual(selection["hard_negative_seed_count"], 1)
        self.assertEqual(selection["excluded_count"], 2)


if __name__ == "__main__":
    unittest.main()
