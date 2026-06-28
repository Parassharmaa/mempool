import json
import tempfile
import unittest
from pathlib import Path

from tools.train_mined_fallback_head import train_and_report


def mined_record(task_id: str, useful: bool) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "bigcodebench_hard",
        "prompt": "filesystem archive task" if useful else "network request task",
        "prompt_features": {
            "categories": ["filesystem"] if useful else ["network"],
            "libraries": ["zipfile"] if useful else ["requests"],
            "missing_libraries": [],
        },
        "source_dataset": "dataset.jsonl",
        "top_worker_id": "qwen",
        "top_probability": 0.55,
        "top_passed": False,
        "top_latency_ms": 10,
        "second_worker_id": "glm",
        "second_probability": 0.45,
        "second_passed": useful,
        "second_latency_ms": 20,
        "first_second_margin": 0.1,
        "fallback_opportunity": True,
        "useful_second_fallback": useful,
        "useful_any_fallback": useful,
        "hard_negative": not useful,
        "best_ranked_alternate_worker_id": "glm" if useful else None,
        "best_ranked_alternate_rank": 2 if useful else None,
        "best_ranked_alternate_probability": 0.45 if useful else None,
        "best_ranked_alternate_latency_ms": 20 if useful else None,
        "additional_latency_to_best_ranked_alternate_ms": 20 if useful else None,
        "total_latency_to_best_ranked_alternate_ms": 30 if useful else None,
        "target_worker_id": "glm" if useful else "qwen",
        "target_worker_passed": useful,
        "solvable_by_any_worker": useful,
        "alternate_count": 3,
        "passed_alternate_count": 1 if useful else 0,
        "alternates": [],
    }


class TrainMinedFallbackHeadToolTest(unittest.TestCase):
    def test_train_and_report_returns_model_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "mined.jsonl"
            records = [
                mined_record("positive", True),
                mined_record("negative-a", False),
                mined_record("negative-b", False),
                mined_record("negative-b", False),
            ]
            dataset.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            report = train_and_report(
                dataset=dataset,
                thresholds=[0.2, 0.5],
                epochs=20,
                learning_rate=0.05,
                l2=0.0,
                label_field="useful_any_fallback",
            )

        self.assertEqual(report["raw_record_count"], 4)
        self.assertEqual(report["record_count"], 3)
        self.assertTrue(report["dedupe_task_id"])
        self.assertEqual(report["positive_count"], 1)
        self.assertEqual(report["model"]["policy"], "mined-fallback-logit-head")
        self.assertIn("leave_one_out", report)


if __name__ == "__main__":
    unittest.main()
