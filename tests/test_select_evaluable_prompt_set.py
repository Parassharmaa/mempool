import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.select_evaluable_prompt_set import select_evaluable_prompt_set


class SelectEvaluablePromptSetTest(unittest.TestCase):
    def test_selects_prompt_payload_with_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks = root / "tasks.json"
            tasks.write_text(
                json.dumps(
                    [
                        {
                            "id": "task-a",
                            "prompt": "Use os to inspect files",
                            "family": "bigcodebench_hard",
                            "function_name": "task_func",
                            "tests": ["assert True"],
                        },
                        {
                            "id": "task-b",
                            "prompt": "Use re to parse text",
                            "family": "bigcodebench_hard",
                            "function_name": "task_func",
                            "tests": ["assert True"],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "prompts.json"

            def fake_predict(*, model_path, record):
                worker = "worker-a" if record["task_id"] == "task-a" else "worker-b"
                return {
                    "selected_worker_id": worker,
                    "worker_distribution": {worker: 0.9, "other": 0.1},
                }

            with patch("tools.select_evaluable_prompt_set.predict_orchestration", side_effect=fake_predict):
                payload = select_evaluable_prompt_set(
                    task_file=tasks,
                    model_path=root / "model.json",
                    output_path=output,
                    limit=2,
                )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["prompt_count"], 2)
        self.assertEqual(saved["prompts"][0]["task_id"], "task-a")
        self.assertEqual(saved["prompts"][1]["predicted_worker_id"], "worker-b")
        self.assertIn("categories", saved["prompts"][0])

    def test_prefers_distinct_predicted_workers_before_filling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks = root / "tasks.json"
            tasks.write_text(
                json.dumps(
                    [
                        {
                            "id": "task-a",
                            "prompt": "Use os to inspect files",
                            "family": "bigcodebench_hard",
                            "function_name": "task_func",
                            "tests": ["assert True"],
                        },
                        {
                            "id": "task-b",
                            "prompt": "Use pathlib to inspect files",
                            "family": "bigcodebench_hard",
                            "function_name": "task_func",
                            "tests": ["assert True"],
                        },
                        {
                            "id": "task-c",
                            "prompt": "Use re to parse text",
                            "family": "bigcodebench_hard",
                            "function_name": "task_func",
                            "tests": ["assert True"],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            def fake_predict(*, model_path, record):
                worker = "worker-a" if record["task_id"] in {"task-a", "task-b"} else "worker-c"
                confidence = 0.95 if record["task_id"] == "task-a" else 0.9
                return {
                    "selected_worker_id": worker,
                    "worker_distribution": {worker: confidence, "other": round(1.0 - confidence, 4)},
                }

            with patch("tools.select_evaluable_prompt_set.predict_orchestration", side_effect=fake_predict):
                payload = select_evaluable_prompt_set(
                    task_file=tasks,
                    model_path=root / "model.json",
                    output_path=root / "prompts.json",
                    limit=2,
                )

        self.assertEqual([item["task_id"] for item in payload["prompts"]], ["task-a", "task-c"])


if __name__ == "__main__":
    unittest.main()
