import json
import tempfile
import unittest
from pathlib import Path

from mempool.qwen_logits_orchestrator import (
    QwenLogitsTrainingConfig,
    audit_qwen_training_readiness,
    build_qwen_logits_training_plan,
    decision_text,
    target_summary,
)


def example() -> dict:
    target = {
        "worker_distribution": {"glm": 0.7, "qwen": 0.3},
        "workflow_distribution": {"direct": 1.0, "verify_then_fallback": 0.0},
        "workflow_kind": "direct",
        "verifier_probability": 0.2,
        "abstain_probability": 0.0,
    }
    return {
        "schema_version": "mempool.small_orchestrator_substrate.v1",
        "task_id": "task-1",
        "benchmark_id": "bench",
        "task_family": "bigcodebench_hard",
        "dense_features": {"bias": 1.0},
        "workers": [{"worker_id": "glm"}, {"worker_id": "qwen"}],
        "target": target,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "route this task"},
            {"role": "assistant", "content": json.dumps(target)},
        ],
    }


class QwenLogitsOrchestratorTest(unittest.TestCase):
    def test_extracts_decision_text_and_target_summary(self) -> None:
        row = example()

        self.assertEqual(decision_text(row), "route this task")
        self.assertEqual(target_summary(row)["worker_distribution"]["glm"], 0.7)

    def test_builds_training_plan_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            substrate = root / "substrate.jsonl"
            substrate.write_text(json.dumps(example()) + "\n", encoding="utf-8")
            plan_path = root / "plan.json"
            rows_path = root / "rows.jsonl"

            plan = build_qwen_logits_training_plan(
                substrate_path=substrate,
                output_path=plan_path,
                rows_output_path=rows_path,
                config=QwenLogitsTrainingConfig(base_model="Qwen/test"),
            )
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(plan["record_count"], 1)
        self.assertEqual(plan["worker_ids"], ["glm", "qwen"])
        self.assertEqual(rows[0]["text"], "route this task")
        self.assertIn("torch", plan["dependency_status"])
        self.assertIn("freeze Qwen-small backbone", plan["training_order"])

    def test_audits_training_readiness(self) -> None:
        report = audit_qwen_training_readiness(backend="transformers")

        self.assertEqual(report["schema_version"], "mempool.qwen_training_readiness.v1")
        self.assertIn("python_version", report)
        self.assertIn("torch", report["dependency_status"])
        self.assertIsInstance(report["recommendations"], list)


if __name__ == "__main__":
    unittest.main()
