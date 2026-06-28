import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tools.run_real_smoke_benchmark import (
    evaluator_environment,
    load_env_file,
    load_existing_records,
    missing_required_packages,
    rebuild_summary,
    record_key,
    run_task,
)
from mempool.smoke_benchmark import SmokeCodeTask


class RaisingClient:
    def chat(self, **kwargs):
        raise TimeoutError("read operation timed out")


class RealSmokeResumeTest(unittest.TestCase):
    def test_load_env_file_sets_missing_values_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "NEW_KEY='new-value'",
                        'EXISTING_KEY="from-file"',
                        "MALFORMED",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"EXISTING_KEY": "from-env"}, clear=True):
                loaded = load_env_file(path)

                self.assertEqual(loaded, ["NEW_KEY"])
                self.assertEqual(os.environ["NEW_KEY"], "new-value")
                self.assertEqual(os.environ["EXISTING_KEY"], "from-env")

    def test_loads_existing_records_by_worker_and_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            path.write_text(
                """
                {
                  "workers": [
                    {
                      "records": [
                        {
                          "worker_id": "w1",
                          "task": {"id": "t1"},
                          "result": {"passed": true}
                        }
                      ]
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            records = load_existing_records(path)

        self.assertIn(record_key("w1", "t1"), records)
        self.assertTrue(records[record_key("w1", "t1")]["result"]["passed"])

    def test_rebuild_summary_counts_existing_records(self) -> None:
        config = {
            "base_url": "http://example.test/v1",
            "workers": [
                {"id": "cheap", "model": "cheap-model"},
                {"id": "strong", "model": "strong-model"},
            ],
        }
        task_id = "smoke-add-numbers"
        existing_records = {
            record_key("cheap", task_id): {
                "worker_id": "cheap",
                "model": "cheap-model",
                "task": {"id": task_id, "family": "code_easy"},
                "result": {"passed": True, "score": 1.0, "failure_mode": None},
                "cost_usd": 0.0,
                "latency_ms": 10,
            },
            record_key("strong", task_id): {
                "worker_id": "strong",
                "model": "strong-model",
                "task": {"id": task_id, "family": "code_easy"},
                "result": {"passed": False, "score": 0.0, "failure_mode": "wrong_answer"},
                "cost_usd": 0.0,
                "latency_ms": 20,
            },
        }

        summary = rebuild_summary(
            config=config,
            tasks_path=Path("research/evals/smoke_code_tasks.json"),
            run_id="resume-test",
            limit=1,
            repeat_count=1,
            existing_records=existing_records,
            timestamp="2026-06-27T00:00:00+00:00",
            evaluator_env={"python_executable": "/x/python", "python_version": "3.x"},
        )

        self.assertEqual(summary["evaluator_env"]["python_executable"], "/x/python")
        self.assertEqual(summary["workers"][0]["solved"], 1)
        self.assertEqual(summary["workers"][0]["pass_at_1"], 1.0)
        self.assertEqual(summary["workers"][1]["solved"], 0)
        self.assertEqual(summary["workers"][1]["pass_at_1"], 0.0)

    def test_run_task_records_request_timeout(self) -> None:
        task = SmokeCodeTask(
            id="timeout-task",
            prompt="Define add_numbers.",
            family="code_easy",
            function_name="add_numbers",
            tests=("assert add_numbers(1, 2) == 3",),
        )
        worker = {"id": "slow-worker", "model": "slow-model", "cost_usd": 0.0}

        record = run_task(
            client=RaisingClient(),
            worker=worker,
            task=task,
            tasks_path=Path("research/evals/smoke_code_tasks.json"),
            eval_timeout_seconds=1,
        )

        self.assertFalse(record["result"]["passed"])
        self.assertEqual(record["result"]["failure_mode"], "request_timeout")
        self.assertEqual(record["result"]["metadata"]["error_type"], "TimeoutError")
        self.assertEqual(record["raw_output"], "")
        self.assertEqual(record["extracted_code"], "")
        self.assertEqual(record["sample_index"], 0)

    def test_rebuild_summary_keeps_repeated_samples(self) -> None:
        config = {
            "base_url": "http://example.test/v1",
            "workers": [{"id": "w1", "model": "m1"}],
        }
        task_id = "smoke-add-numbers"
        existing_records = {
            record_key("w1", task_id, 0): {
                "worker_id": "w1",
                "model": "m1",
                "sample_index": 0,
                "task": {"id": task_id, "family": "code_easy"},
                "result": {"passed": True, "score": 1.0, "failure_mode": None},
                "cost_usd": 0.0,
                "latency_ms": 10,
            },
            record_key("w1", task_id, 1): {
                "worker_id": "w1",
                "model": "m1",
                "sample_index": 1,
                "task": {"id": task_id, "family": "code_easy"},
                "result": {"passed": False, "score": 0.0, "failure_mode": "test_failure"},
                "cost_usd": 0.0,
                "latency_ms": 20,
            },
        }

        summary = rebuild_summary(
            config=config,
            tasks_path=Path("research/evals/smoke_code_tasks.json"),
            run_id="repeat-test",
            limit=1,
            repeat_count=2,
            existing_records=existing_records,
            timestamp="2026-06-27T00:00:00+00:00",
        )

        self.assertEqual(summary["workers"][0]["task_count"], 2)
        self.assertEqual(summary["workers"][0]["solved"], 1)
        self.assertEqual(summary["workers"][0]["pass_at_1"], 0.5)

    def test_evaluator_environment_reports_required_packages(self) -> None:
        env = evaluator_environment(["json", "definitely_missing_mempool_package"])

        self.assertTrue(env["required_packages"]["json"])
        self.assertFalse(env["required_packages"]["definitely_missing_mempool_package"])
        self.assertIn("python_executable", env)

    def test_missing_required_packages(self) -> None:
        self.assertEqual(
            missing_required_packages(["json", "definitely_missing_mempool_package"]),
            ["definitely_missing_mempool_package"],
        )


if __name__ == "__main__":
    unittest.main()
