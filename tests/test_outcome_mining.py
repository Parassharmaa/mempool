import unittest
from pathlib import Path

from mempool.outcome_mining import (
    is_broad_pass_latency_row,
    is_exclusive_stable_nonqwen_target,
    is_stable_nonqwen_target,
    rank_outcome_sources,
    summarize_outcome_source,
)


def record(task_id: str, target: str, pass_rates: dict[str, float]) -> dict:
    return {
        "task_id": task_id,
        "target_worker_id": target,
        "workers": [
            {
                "worker_id": worker_id,
                "passed": pass_rate > 0.0,
                "pass_rate": pass_rate,
            }
            for worker_id, pass_rate in pass_rates.items()
        ],
    }


class OutcomeMiningTest(unittest.TestCase):
    def test_identifies_broad_pass_latency_rows(self) -> None:
        broad = record("broad", "ollama-cloud-glm-5.2", {"ollama-cloud-glm-5.2": 1.0, "ollama-cloud-qwen3-coder-480b": 1.0})
        narrow = record("narrow", "ollama-cloud-glm-5.2", {"ollama-cloud-glm-5.2": 1.0, "ollama-cloud-qwen3-coder-480b": 0.0})

        self.assertTrue(is_broad_pass_latency_row(broad))
        self.assertFalse(is_broad_pass_latency_row(narrow))

    def test_identifies_stable_nonqwen_targets(self) -> None:
        nonqwen = record("nonqwen", "ollama-cloud-glm-5.2", {"ollama-cloud-glm-5.2": 1.0, "ollama-cloud-qwen3-coder-480b": 0.0})
        qwen = record("qwen", "ollama-cloud-qwen3-coder-480b", {"ollama-cloud-qwen3-coder-480b": 1.0})
        unstable = record("unstable", "ollama-cloud-glm-5.2", {"ollama-cloud-glm-5.2": 0.5})

        self.assertTrue(is_stable_nonqwen_target(nonqwen))
        self.assertTrue(is_exclusive_stable_nonqwen_target(nonqwen))
        self.assertFalse(is_stable_nonqwen_target(qwen))
        self.assertFalse(is_stable_nonqwen_target(unstable))
        self.assertFalse(
            is_exclusive_stable_nonqwen_target(
                record(
                    "broad",
                    "ollama-cloud-glm-5.2",
                    {
                        "ollama-cloud-glm-5.2": 1.0,
                        "ollama-cloud-qwen3-coder-480b": 1.0,
                    },
                )
            )
        )

    def test_summarizes_and_ranks_sources(self) -> None:
        weak = summarize_outcome_source(
            source_path=Path("weak.jsonl"),
            records=[record("qwen", "ollama-cloud-qwen3-coder-480b", {"ollama-cloud-qwen3-coder-480b": 1.0})],
        )
        strong = summarize_outcome_source(
            source_path=Path("strong.jsonl"),
            records=[
                record(
                    "broad",
                    "ollama-cloud-glm-5.2",
                    {
                        "ollama-cloud-glm-5.2": 1.0,
                        "ollama-cloud-qwen3-coder-480b": 1.0,
                    },
                )
            ],
        )

        ranked = rank_outcome_sources([weak, strong])

        self.assertEqual(ranked[0]["source"], "strong.jsonl")
        self.assertEqual(ranked[0]["stable_nonqwen_targets"], 1)
        self.assertEqual(ranked[0]["exclusive_stable_nonqwen_targets"], 0)
        self.assertEqual(ranked[0]["broad_pass_latency_rows"], 1)


if __name__ == "__main__":
    unittest.main()
