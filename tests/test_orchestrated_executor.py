import json
import tempfile
import unittest
from pathlib import Path

from mempool.multi_head_orchestrator import train_multi_head_orchestrator
from mempool.orchestrated_executor import (
    execute_orchestrated_prompt,
    flatten_orchestrated_execution,
    load_worker_pool,
    write_orchestrated_outcome,
    worker_by_id,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, model: str, messages: list[dict[str, str]]) -> dict:
        self.calls.append({"model": model, "messages": messages})
        return {"choices": [{"message": {"content": f"answered by {model}"}}]}


def example(task_id: str, keyword: str, target_worker: str) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "bigcodebench_hard",
        "prompt": f"Task mentions {keyword}",
        "dense_features": {"bias": 1.0, f"category_{keyword}": 1.0, f"signal_{keyword}": 1.0},
        "target": {
            "worker_distribution": {
                "glm": 0.95 if target_worker == "glm" else 0.05,
                "qwen": 0.95 if target_worker == "qwen" else 0.05,
            },
            "target_worker_id": target_worker,
            "workflow_distribution": {"direct": 1.0, "verify_then_fallback": 0.0},
            "workflow_kind": "direct",
            "verifier_probability": 0.1,
            "abstain_probability": 0.0,
        },
        "workers": [
            {"worker_id": "glm", "pass_rate": 1.0, "mean_latency_ms": 10.0},
            {"worker_id": "qwen", "pass_rate": 1.0, "mean_latency_ms": 20.0},
        ],
    }


class OrchestratedExecutorTest(unittest.TestCase):
    def test_routes_prompt_to_selected_worker_and_records_provenance(self) -> None:
        records = [
            example("t1", "filesystem", "glm"),
            example("t2", "network", "qwen"),
        ]
        model, _ = train_multi_head_orchestrator(records, epochs=80, learning_rate=0.05)
        fake_client = FakeClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model.json"
            model_path.write_text(
                json.dumps(
                    {
                        "model_type": "linear-multi-head-orchestrator",
                        "substrate": "fixture.jsonl",
                        "orchestrator": model.to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            worker_pool = root / "pool.json"
            worker_pool.write_text(
                json.dumps(
                    {
                        "base_url": "http://localhost:11434/v1",
                        "workers": [
                            {"id": "glm", "model": "glm-5.2", "strengths": ["general"]},
                            {"id": "qwen", "model": "qwen3-coder:480b", "strengths": ["code"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = execute_orchestrated_prompt(
                model_path=model_path,
                worker_pool_path=worker_pool,
                prompt="Please solve a filesystem task.",
                task_id="adhoc",
                task_family="bigcodebench_hard",
                categories=["filesystem"],
                client=fake_client,
            )

        self.assertEqual(result["schema_version"], "mempool.orchestrated_execution.v1")
        self.assertEqual(result["route"]["selected_worker_id"], "glm")
        self.assertEqual(result["selected_worker"]["model"], "glm-5.2")
        self.assertEqual(result["response"]["content"], "answered by glm-5.2")
        self.assertEqual(fake_client.calls[0]["model"], "glm-5.2")
        self.assertEqual(fake_client.calls[0]["messages"][-1]["content"], "Please solve a filesystem task.")

    def test_rejects_worker_pool_without_selected_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = Path(tmpdir) / "pool.json"
            pool.write_text(
                json.dumps({"base_url": "http://localhost/v1", "workers": [{"id": "a", "model": "ma"}]}),
                encoding="utf-8",
            )
            payload = load_worker_pool(pool)

        with self.assertRaisesRegex(ValueError, "not present"):
            worker_by_id(payload, "missing")

    def test_dry_run_returns_route_without_calling_client(self) -> None:
        records = [example("t1", "filesystem", "glm")]
        model, _ = train_multi_head_orchestrator(records, epochs=10, learning_rate=0.05)
        fake_client = FakeClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model.json"
            model_path.write_text(
                json.dumps(
                    {
                        "model_type": "linear-multi-head-orchestrator",
                        "substrate": "fixture.jsonl",
                        "orchestrator": model.to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            worker_pool = root / "pool.json"
            worker_pool.write_text(
                json.dumps(
                    {
                        "base_url": "http://localhost:11434/v1",
                        "workers": [{"id": "glm", "model": "glm-5.2"}],
                    }
                ),
                encoding="utf-8",
            )

            result = execute_orchestrated_prompt(
                model_path=model_path,
                worker_pool_path=worker_pool,
                prompt="Please solve a filesystem task.",
                categories=["filesystem"],
                client=fake_client,
                dry_run=True,
            )

        self.assertEqual(result["execution_status"], "dry_run")
        self.assertEqual(result["response"]["content"], "")
        self.assertEqual(fake_client.calls, [])

    def test_flattens_orchestrated_execution_as_outcome_row(self) -> None:
        result = {
            "timestamp": "2026-06-28T00:00:00+00:00",
            "model_path": "model.json",
            "worker_pool_path": "pool.json",
            "task_id": "adhoc",
            "benchmark_id": "local-ad-hoc",
            "task_family": "bigcodebench_hard",
            "prompt": "do work",
            "route": {
                "selected_workflow": "direct",
                "worker_distribution": {"glm": 0.7, "qwen": 0.3},
                "workflow_distribution": {"direct": 1.0},
                "verifier_probability": 0.2,
                "abstain_probability": 0.1,
            },
            "selected_worker": {"id": "glm", "model": "glm-5.2", "cost_usd": 0.0},
            "response": {"content": "answer"},
            "latency_ms": 123,
            "execution_status": "completed",
        }

        row = flatten_orchestrated_execution(result)

        self.assertEqual(row["schema_version"], "mempool.orchestrated_execution_outcome.v1")
        self.assertEqual(row["selected_worker_id"], "glm")
        self.assertEqual(row["selected_model"], "glm-5.2")
        self.assertEqual(row["response_chars"], 6)
        self.assertTrue(row["response_present"])
        self.assertIsNone(row["passed"])

    def test_writes_orchestrated_outcome_jsonl(self) -> None:
        result = {
            "task_id": "adhoc",
            "benchmark_id": "local-ad-hoc",
            "task_family": "ad_hoc",
            "prompt": "hello",
            "route": {},
            "selected_worker": {"id": "glm", "model": "glm-5.2"},
            "response": {"content": ""},
            "execution_status": "dry_run",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "outcome.jsonl"
            row = write_orchestrated_outcome(output, result)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(row["execution_status"], "dry_run")
        self.assertEqual(rows[0]["selected_worker_id"], "glm")


if __name__ == "__main__":
    unittest.main()
