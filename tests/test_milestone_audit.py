import json
import tempfile
import unittest
from pathlib import Path

from mempool.milestone_audit import apply_milestone_audit, audit_milestones


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def routing_row(task_id: str, target: str) -> dict:
    return {
        "task_id": task_id,
        "target_worker_id": target,
        "workers": [
            {"worker_id": "w1", "passed": target == "w1"},
            {"worker_id": "w2", "passed": target == "w2"},
            {"worker_id": "w3", "passed": False},
        ],
    }


class MilestoneAuditTest(unittest.TestCase):
    def test_audit_reports_completed_core_and_quarantined_m5(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            routing = write_jsonl(
                root / "routing.jsonl",
                [routing_row("t1", "w1"), routing_row("t2", "w2")],
            )
            active = write_json(
                root / "active.json",
                {
                    "active": {
                        "model": "model.json",
                        "dataset": str(routing),
                        "loo": {
                            "available": True,
                            "task_count": 10,
                            "solvable_pass_at_1": 0.8,
                        },
                        "target_mix": {
                            "task_count": 10,
                            "target_worker_count": 2,
                        },
                    }
                },
            )
            write_json(root / "model.json", {"model_type": "test"})
            write_json(root / "external_report.json", {"ok": True})
            write_json(root / "comparison.json", {"ok": True})
            write_json(root / "m5_model.json", {"ok": True})
            write_json(
                root / "m5_report.json",
                {
                    "evaluation": {
                        "task_count": 50,
                        "target_accuracy": 0.74,
                        "workflow_accuracy": 0.94,
                    },
                    "leave_one_out": {
                        "task_count": 50,
                        "target_accuracy": 0.62,
                        "mean_latency_regret_ms": 3609.6,
                    },
                },
            )
            write_json(
                root / "m5_gate.json",
                {"decision": "quarantine", "reasons": ["too much latency regret"]},
            )
            write_json(root / "tb_report.json", {"ok": True})
            write_jsonl(root / "tb_traj.jsonl", [{"task_success": False}])
            write_json(
                root / "refresh.json",
                {
                    "decision": "quarantine",
                    "promotion_profile": "preserve_accuracy",
                    "guardrails": [{"name": "rollback", "passed": True}],
                    "promotion": {"allowed": False},
                },
            )

            report = audit_milestones(
                root=root,
                evidence={
                    "worker_outcomes": "missing.jsonl",
                    "routing_dataset": str(routing),
                    "active_policy": str(active),
                    "external_smoke_report": "external_report.json",
                    "external_comparison_report": "comparison.json",
                    "m5_model": "m5_model.json",
                    "m5_report": "m5_report.json",
                    "m5_gate": "m5_gate.json",
                    "terminal_bench_report": "tb_report.json",
                    "terminal_bench_trajectories": "tb_traj.jsonl",
                    "adaptive_refresh": "refresh.json",
                },
            )

        statuses = {
            milestone["id"]: milestone["audit_status"]
            for milestone in report["milestones"]
        }
        self.assertEqual(statuses["M3-lightweight-router"], "completed")
        self.assertEqual(statuses["M4-external-smoke-benchmark"], "completed")
        self.assertEqual(statuses["M5-small-trainable-orchestrator"], "completed")
        m6 = next(
            milestone
            for milestone in report["milestones"]
            if milestone["id"] == "M6-adaptive-memory-refresh"
        )
        self.assertEqual(m6["metrics"]["promotion_profile"], "preserve_accuracy")
        self.assertTrue(any("preserve_accuracy" in gap for gap in m6["open_gaps"]))
        m5 = next(
            milestone
            for milestone in report["milestones"]
            if milestone["id"] == "M5-small-trainable-orchestrator"
        )
        self.assertTrue(any("quarantined" in gap for gap in m5["open_gaps"]))

    def test_apply_milestone_audit_updates_active_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            milestones = write_json(
                root / "milestones.json",
                {
                    "active_milestone": "old",
                    "milestones": [
                        {"id": "M1-real-worker-pool-evaluation", "status": "pending"},
                        {"id": "M5.5-agentic-harness-pilot", "status": "pending"},
                    ],
                },
            )
            audit = {
                "schema_version": "test",
                "recommended_active_milestone": "M5.5-agentic-harness-pilot",
                "milestones": [
                    {
                        "id": "M1-real-worker-pool-evaluation",
                        "audit_status": "completed",
                    },
                    {
                        "id": "M5.5-agentic-harness-pilot",
                        "audit_status": "partial",
                    },
                ],
            }

            updated = apply_milestone_audit(milestones, audit)

        self.assertEqual(updated["active_milestone"], "M5.5-agentic-harness-pilot")
        self.assertEqual(updated["milestones"][0]["status"], "completed")
        self.assertEqual(updated["milestones"][1]["status"], "active")


if __name__ == "__main__":
    unittest.main()
