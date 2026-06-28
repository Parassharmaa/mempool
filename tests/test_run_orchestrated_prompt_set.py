import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.run_orchestrated_prompt_set import run_prompt_set


def fake_execution(**kwargs):
    worker_id = kwargs.get("fixed_worker_id") or "router-worker"
    return {
        "schema_version": "mempool.orchestrated_execution.v1",
        "timestamp": "2026-06-28T00:00:00+00:00",
        "model_path": str(kwargs["model_path"]),
        "worker_pool_path": str(kwargs["worker_pool_path"]),
        "policy_id": f"fixed-worker:{worker_id}" if kwargs.get("fixed_worker_id") else "trained-orchestrator",
        "fixed_worker_id": kwargs.get("fixed_worker_id"),
        "task_id": kwargs["task_id"],
        "benchmark_id": kwargs["benchmark_id"],
        "task_family": kwargs["task_family"],
        "prompt": kwargs["prompt"],
        "route": {
            "selected_worker_id": worker_id,
            "selected_workflow": "direct",
            "worker_distribution": {worker_id: 1.0},
            "workflow_distribution": {"direct": 1.0},
            "verifier_probability": 0.0,
            "abstain_probability": 0.0,
        },
        "selected_worker": {"id": worker_id, "model": f"{worker_id}-model", "cost_usd": 0.0},
        "request": {"messages": [], "chat_options": {}},
        "response": {"content": "ok", "raw": {}},
        "latency_ms": 1,
        "execution_status": "completed",
    }


class RunOrchestratedPromptSetTest(unittest.TestCase):
    def test_runs_orchestrated_and_fixed_rows_for_each_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_set = root / "prompts.json"
            prompt_set.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {"task_id": "one", "prompt": "hello", "categories": ["general"]},
                            {"task_id": "two", "prompt": "files", "categories": ["filesystem"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "summary.json"
            outcomes = root / "outcomes.jsonl"

            with patch("tools.run_orchestrated_prompt_set.execute_orchestrated_prompt", side_effect=fake_execution), patch(
                "tools.run_orchestrated_prompt_set.execute_fixed_worker_prompt",
                side_effect=fake_execution,
            ):
                summary = run_prompt_set(
                    prompt_set=prompt_set,
                    model=root / "model.json",
                    worker_pool=root / "pool.json",
                    fixed_worker_id="fixed",
                    output=output,
                    outcomes_output=outcomes,
                    env_file=None,
                    dry_run=False,
                )
            rows = [json.loads(line) for line in outcomes.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["prompt_count"], 2)
        self.assertEqual(summary["execution_count"], 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["policy_id"], "trained-orchestrator")
        self.assertEqual(rows[1]["policy_id"], "fixed-worker:fixed")


if __name__ == "__main__":
    unittest.main()
