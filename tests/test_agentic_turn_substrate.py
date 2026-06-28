import json
import tempfile
import unittest
from pathlib import Path

from mempool.agentic_turn_substrate import (
    build_agentic_turn_examples,
    build_agentic_turn_substrate,
)


def trajectory(task_success: bool = True) -> dict:
    return {
        "benchmark_id": "terminal-bench-2.1",
        "run_id": "tb-pilot",
        "task_id": "fix-git",
        "trial_id": "trial-1",
        "agent_id": "single-worker-shell-agent",
        "worker_id": "qwen",
        "policy_id": "fixed-worker-qwen",
        "selected_workflow": "terminal-agent",
        "task_success": task_success,
        "verifier_passed": task_success,
        "latency_ms": 1200,
        "cost_usd": 0.01,
        "terminal_actions": [
            {
                "index": 0,
                "action_kind": "inspect",
                "summary": "inspected repository files",
                "exit_code": 0,
            },
            {
                "index": 1,
                "action_kind": "test",
                "summary": "ran summarized verifier",
                "exit_code": 0 if task_success else 1,
            },
        ],
        "file_edits": [
            {"path": "solution.py", "operation": "modify"}
        ],
        "tests_run": [
            {"command_summary": "ran task verifier", "passed": task_success}
        ],
        "worker_switches": [],
        "failure_mode": None if task_success else "test_failure",
    }


class AgenticTurnSubstrateTest(unittest.TestCase):
    def test_builds_one_turn_example_per_sanitized_action(self) -> None:
        examples = build_agentic_turn_examples([trajectory()])

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["schema_version"], "mempool.agentic_turn_substrate.v1")
        self.assertEqual(examples[0]["target"]["target_worker_id"], "qwen")
        self.assertEqual(examples[0]["target"]["action_kind"], "inspect")
        self.assertEqual(examples[1]["target"]["action_kind"], "test")
        self.assertEqual(examples[1]["target"]["stop_probability"], 1.0)
        self.assertEqual(examples[1]["target"]["memory_update_probability"], 1.0)
        self.assertEqual(json.loads(examples[1]["messages"][-1]["content"]), examples[1]["target"])

    def test_marks_failed_final_turn_as_repair_without_memory_update(self) -> None:
        examples = build_agentic_turn_examples([trajectory(task_success=False)])

        self.assertEqual(examples[-1]["target"]["repair_probability"], 1.0)
        self.assertEqual(examples[-1]["target"]["stop_probability"], 0.0)
        self.assertEqual(examples[-1]["target"]["memory_update_probability"], 0.0)
        self.assertEqual(examples[-1]["dense_features"]["test_failure_seen"], 1.0)

    def test_rejects_raw_terminal_content(self) -> None:
        row = trajectory()
        row["terminal_actions"][0]["stdout"] = "raw output should not be persisted"

        with self.assertRaisesRegex(ValueError, "forbidden content fields"):
            build_agentic_turn_examples([row])

    def test_writes_jsonl_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectories = root / "trajectories.jsonl"
            trajectories.write_text(json.dumps(trajectory()) + "\n", encoding="utf-8")
            output = root / "turns.jsonl"
            manifest_path = root / "manifest.json"

            manifest = build_agentic_turn_substrate(
                trajectory_path=trajectories,
                output_path=output,
                manifest_path=manifest_path,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(manifest["record_count"], 2)
        self.assertEqual(manifest["trajectory_count"], 1)
        self.assertEqual(manifest["training_status"], "schema_ready_not_trained")
        self.assertEqual(manifest["target_worker_counts"], {"qwen": 2})
        self.assertEqual(rows[0]["turn_id"], "trial-1:0")


if __name__ == "__main__":
    unittest.main()
