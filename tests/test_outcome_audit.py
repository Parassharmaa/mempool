import unittest

from mempool.outcome_audit import audit_outcome_rows


def row(task_id: str = "task", worker_id: str = "worker") -> dict:
    return {
        "benchmark_id": "bench",
        "run_id": "run",
        "timestamp": "2026-06-27T00:00:00+00:00",
        "task_id": task_id,
        "task_family": "code",
        "prompt": "Write code.",
        "worker_id": worker_id,
        "model": worker_id,
        "workflow_kind": "route",
        "passed": True,
        "score": 1.0,
        "failure_mode": None,
        "latency_ms": 10,
        "cost_usd": 0.0,
        "reward": 1.0,
        "evaluator_required_packages": {"numpy": True},
    }


class OutcomeAuditTest(unittest.TestCase):
    def test_ready_when_fields_packages_and_coverage_match(self) -> None:
        report = audit_outcome_rows(
            [row(worker_id="a"), row(worker_id="b")],
            required_evaluator_packages=["numpy"],
            min_workers_per_task=2,
        )

        self.assertTrue(report["ready_for_conversion"])
        self.assertEqual(report["package_mismatch_rows"], 0)

    def test_rejects_missing_package_and_missing_fields(self) -> None:
        bad = row()
        bad.pop("latency_ms")
        bad["evaluator_required_packages"] = {"numpy": False}

        report = audit_outcome_rows([bad], required_evaluator_packages=["numpy"])

        self.assertFalse(report["ready_for_conversion"])
        self.assertEqual(report["missing_field_counts"], {"latency_ms": 1})
        self.assertEqual(report["package_mismatch_rows"], 1)

    def test_rejects_low_worker_or_sample_coverage(self) -> None:
        report = audit_outcome_rows(
            [row(task_id="task-a", worker_id="a")],
            min_workers_per_task=2,
            min_samples_per_worker_task=2,
        )

        self.assertFalse(report["ready_for_conversion"])
        self.assertEqual(report["underspecified_tasks"], ["task-a"])
        self.assertEqual(
            report["low_sample_pairs"],
            [{"task_id": "task-a", "worker_id": "a", "samples": 1}],
        )


if __name__ == "__main__":
    unittest.main()
