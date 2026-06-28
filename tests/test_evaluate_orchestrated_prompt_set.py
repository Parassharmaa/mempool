import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_orchestrated_prompt_set import evaluate_prompt_set


class EvaluateOrchestratedPromptSetTest(unittest.TestCase):
    def test_scores_execution_rows_against_task_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks = root / "tasks.json"
            tasks.write_text(
                json.dumps(
                    [
                        {
                            "id": "task-1",
                            "prompt": "Return one",
                            "family": "unit",
                            "function_name": "task_func",
                            "tests": ["assert task_func() == 1"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "executions": [
                            {
                                "task_id": "task-1",
                                "policy_id": "trained-orchestrator",
                                "selected_worker": {"id": "w1", "model": "m1"},
                                "response": {"content": "```python\ndef task_func():\n    return 1\n```"},
                                "latency_ms": 5,
                            },
                            {
                                "task_id": "task-1",
                                "policy_id": "fixed-worker:w2",
                                "selected_worker": {"id": "w2", "model": "m2"},
                                "response": {"content": "def task_func():\n    return 2"},
                                "latency_ms": 7,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "rows.jsonl"
            report = root / "report.json"

            payload = evaluate_prompt_set(
                comparison_path=comparison,
                task_paths=[tasks],
                output_path=output,
                report_path=report,
                timeout_seconds=2,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(payload["record_count"], 2)
        self.assertTrue(rows[0]["passed"])
        self.assertFalse(rows[1]["passed"])
        by_policy = {row["policy_id"]: row for row in payload["policy_summaries"]}
        self.assertEqual(by_policy["trained-orchestrator"]["pass_rate"], 1.0)
        self.assertEqual(by_policy["fixed-worker:w2"]["pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
