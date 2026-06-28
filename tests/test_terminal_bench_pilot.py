import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mempool.terminal_bench import (
    compare_terminal_bench_trajectories,
    extract_terminal_bench_metadata,
    harbor_job_to_terminal_bench_trajectories,
    select_terminal_bench_pilot,
    summarize_harbor_job,
    validate_terminal_bench_trajectories,
)


class TerminalBenchPilotTest(unittest.TestCase):
    def test_selects_diverse_metadata_only_subset(self) -> None:
        rows = [
            {
                "id": "tb-system-hard",
                "categories": ["system-administration"],
                "difficulty": "hard",
            },
            {
                "id": "tb-security-medium",
                "categories": ["security"],
                "difficulty": "medium",
            },
            {
                "id": "tb-data-easy",
                "categories": ["data-processing"],
                "difficulty": "easy",
            },
        ]

        manifest = select_terminal_bench_pilot(
            rows,
            limit=2,
            preferred_categories=("security", "system-administration"),
        )

        self.assertEqual(manifest["benchmark_id"], "terminal-bench-2.1")
        self.assertEqual(
            manifest["selected_task_ids"],
            ["tb-security-medium", "tb-system-hard"],
        )
        self.assertIn("run_contract", manifest)

    def test_rejects_task_content_fields(self) -> None:
        rows = [
            {
                "id": "tb-with-prompt",
                "category": "software-engineering",
                "prompt": "do the private benchmark task",
            }
        ]

        with self.assertRaisesRegex(ValueError, "forbidden content fields"):
            select_terminal_bench_pilot(rows, limit=1)

    def test_extracts_metadata_from_export_without_content_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(
                """[
  {
    "id": "tb-safe",
    "category": "software-engineering",
    "difficulty": "hard",
    "tags": ["git"],
    "prompt": "private task prompt",
    "tests": "private verifier"
  }
]""",
                encoding="utf-8",
            )

            rows = extract_terminal_bench_metadata([path])

        self.assertEqual(rows[0]["id"], "tb-safe")
        self.assertEqual(rows[0]["category"], "software-engineering")
        self.assertNotIn("prompt", rows[0])
        self.assertNotIn("tests", rows[0])

    def test_scans_task_directories_as_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = root / "tasks" / "security" / "crack-hash"
            task_dir.mkdir(parents=True)
            (task_dir / "task.yaml").write_text("instruction: do not copy\n", encoding="utf-8")

            rows = extract_terminal_bench_metadata([root])

        self.assertEqual(rows[0]["id"], "tasks/security/crack-hash")
        self.assertEqual(rows[0]["categories"], ["security"])
        self.assertEqual(rows[0]["difficulty"], "medium")

    def test_scans_task_toml_metadata_without_description(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = root / "tasks" / "configure-git-webserver"
            task_dir.mkdir(parents=True)
            environment_dir = task_dir / "environment"
            environment_dir.mkdir()
            (environment_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
            (task_dir / "task.toml").write_text(
                """
[task]
name = "terminal-bench/configure-git-webserver"
description = "do not persist this task description"
keywords = ["system", "web"]

[metadata]
difficulty = "hard"
category = "system-administration"
tags = ["system", "version-control"]
""",
                encoding="utf-8",
            )

            rows = extract_terminal_bench_metadata([root])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "terminal-bench/configure-git-webserver")
        self.assertEqual(rows[0]["category"], "system-administration")
        self.assertEqual(rows[0]["difficulty"], "hard")
        self.assertEqual(rows[0]["tags"], ["system", "version-control"])
        self.assertNotIn("description", rows[0])

    def test_validates_summary_only_trajectory(self) -> None:
        records = [
            {
                "benchmark_id": "terminal-bench-2.1",
                "run_id": "tb-pilot",
                "task_id": "tb-system-hard",
                "trial_id": "trial-0",
                "agent_id": "single-worker-shell-agent",
                "worker_id": "ollama-cloud-qwen3-coder-480b",
                "policy_id": "fixed-worker",
                "selected_workflow": "terminal-agent",
                "task_success": True,
                "verifier_passed": True,
                "latency_ms": 1200,
                "cost_usd": 0.01,
                "terminal_actions": [
                    {
                        "index": 0,
                        "action_kind": "shell",
                        "summary": "inspected repository files",
                        "exit_code": 0,
                    }
                ],
                "file_edits": [
                    {"path": "solution.py", "operation": "modify"}
                ],
                "tests_run": [
                    {"command_summary": "ran task verifier", "passed": True}
                ],
                "worker_switches": [],
                "failure_mode": None,
            }
        ]

        self.assertEqual(validate_terminal_bench_trajectories(records), [])

    def test_rejects_raw_terminal_content_in_trajectory(self) -> None:
        records = [
            {
                "benchmark_id": "terminal-bench-2.1",
                "run_id": "tb-pilot",
                "task_id": "tb-system-hard",
                "trial_id": "trial-0",
                "agent_id": "single-worker-shell-agent",
                "worker_id": "ollama-cloud-qwen3-coder-480b",
                "policy_id": "fixed-worker",
                "selected_workflow": "terminal-agent",
                "task_success": False,
                "verifier_passed": False,
                "latency_ms": 1200,
                "cost_usd": 0.01,
                "terminal_actions": [
                    {
                        "index": 0,
                        "action_kind": "shell",
                        "summary": "ran a command",
                        "exit_code": 1,
                        "stdout": "raw benchmark output",
                    }
                ],
                "file_edits": [],
                "tests_run": [],
                "worker_switches": [],
                "failure_mode": "test_failure",
            }
        ]

        errors = validate_terminal_bench_trajectories(records)
        self.assertTrue(any("forbidden content fields" in error for error in errors))

    def test_summarizes_complete_harbor_job_without_logs(self) -> None:
        with TemporaryDirectory() as directory:
            job_dir = Path(directory) / "job"
            trial_dir = job_dir / "task__abc"
            trial_dir.mkdir(parents=True)
            (job_dir / "config.json").write_text(
                '{"environment": {"type": "docker"}, "agent": {"type": "oracle"}}',
                encoding="utf-8",
            )
            (job_dir / "result.json").write_text(
                """{
  "id": "job-1",
  "started_at": "2026-06-27T00:00:00Z",
  "updated_at": "2026-06-27T00:01:00Z",
  "finished_at": "2026-06-27T00:01:00Z",
  "n_total_trials": 1,
  "stats": {
    "n_completed_trials": 1,
    "n_errored_trials": 0,
    "n_cancelled_trials": 0,
    "n_running_trials": 0,
    "n_pending_trials": 0,
    "cost_usd": 0.0
  }
}""",
                encoding="utf-8",
            )
            (trial_dir / "config.json").write_text("{}", encoding="utf-8")
            (trial_dir / "trial.log").write_text("raw log not read", encoding="utf-8")

            summary = summarize_harbor_job(job_dir)

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["environment_type"], "docker")
        self.assertEqual(summary["agent_type"], "oracle")
        self.assertEqual(summary["trial_directory_count"], 1)
        self.assertEqual(summary["raw_log_policy"], "not_read")

    def test_summarizes_stale_running_harbor_job(self) -> None:
        with TemporaryDirectory() as directory:
            job_dir = Path(directory) / "job"
            job_dir.mkdir()
            (job_dir / "result.json").write_text(
                """{
  "id": "job-2",
  "started_at": "2026-06-27T00:00:00Z",
  "updated_at": "2026-06-27T00:02:00Z",
  "finished_at": null,
  "n_total_trials": 1,
  "stats": {
    "n_completed_trials": 0,
    "n_errored_trials": 0,
    "n_cancelled_trials": 0,
    "n_running_trials": 1,
    "n_pending_trials": 0
  }
}""",
                encoding="utf-8",
            )

            summary = summarize_harbor_job(job_dir)

        self.assertEqual(summary["status"], "running_or_stale")
        self.assertEqual(summary["n_running_trials"], 1)

    def test_summarizes_interrupted_ambiguous_harbor_job(self) -> None:
        with TemporaryDirectory() as directory:
            job_dir = Path(directory) / "job"
            job_dir.mkdir()
            (job_dir / "result.json").write_text(
                """{
  "id": "job-3",
  "finished_at": null,
  "n_total_trials": 1,
  "stats": {
    "n_completed_trials": 1,
    "n_errored_trials": 1,
    "n_cancelled_trials": 1,
    "n_running_trials": 0,
    "n_pending_trials": 0
  }
}""",
                encoding="utf-8",
            )

            summary = summarize_harbor_job(job_dir)

        self.assertEqual(summary["status"], "interrupted_ambiguous")

    def test_converts_harbor_trial_result_to_safe_trajectory(self) -> None:
        with TemporaryDirectory() as directory:
            job_dir = Path(directory) / "job"
            trial_dir = job_dir / "fix-git__abc"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(
                """{
  "id": "trial-1",
  "started_at": "2026-06-27T12:00:00Z",
  "finished_at": "2026-06-27T12:00:02.500000Z",
  "task_id": {"path": "external_repos/terminal-bench-2-1/tasks/fix-git"},
  "agent_result": {"cost_usd": null},
  "verifier_result": {"rewards": {"reward": 1.0}}
}""",
                encoding="utf-8",
            )

            records = harbor_job_to_terminal_bench_trajectories(
                job_dir,
                run_id="run",
                agent_id="oracle",
                worker_id="oracle",
                policy_id="fixed-oracle",
            )

        self.assertEqual(validate_terminal_bench_trajectories(records), [])
        self.assertEqual(records[0]["task_id"], "fix-git")
        self.assertTrue(records[0]["task_success"])
        self.assertEqual(records[0]["latency_ms"], 2500.0)
        self.assertEqual(records[0]["raw_log_policy"], "not_read")

    def test_compares_terminal_bench_trajectories_by_policy(self) -> None:
        records = [
            {
                "benchmark_id": "terminal-bench-2.1",
                "run_id": "run",
                "task_id": "fix-git",
                "trial_id": "oracle",
                "agent_id": "oracle",
                "worker_id": "oracle-reference",
                "policy_id": "fixed-oracle",
                "selected_workflow": "terminal-agent",
                "task_success": True,
                "verifier_passed": True,
                "latency_ms": 1000,
                "cost_usd": 0,
                "terminal_actions": [],
                "file_edits": [],
                "tests_run": [{"command_summary": "verifier", "passed": True}],
                "worker_switches": [],
                "failure_mode": None,
            },
            {
                "benchmark_id": "terminal-bench-2.1",
                "run_id": "run",
                "task_id": "fix-git",
                "trial_id": "worker",
                "agent_id": "terminus",
                "worker_id": "ollama-cloud-qwen3-coder-next",
                "policy_id": "fixed-worker-qwen-next",
                "selected_workflow": "terminal-agent",
                "task_success": False,
                "verifier_passed": False,
                "latency_ms": 2000,
                "cost_usd": 0,
                "terminal_actions": [],
                "file_edits": [],
                "tests_run": [{"command_summary": "verifier", "passed": False}],
                "worker_switches": [],
                "failure_mode": "verifier_failed",
            },
        ]

        report = compare_terminal_bench_trajectories(records)

        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["task_count"], 1)
        by_policy = {row["policy_id"]: row for row in report["policy_summaries"]}
        self.assertEqual(by_policy["fixed-oracle"]["success_rate"], 1.0)
        self.assertEqual(by_policy["fixed-worker-qwen-next"]["success_rate"], 0.0)
        self.assertEqual(report["task_comparisons"][0]["task_id"], "fix-git")


if __name__ == "__main__":
    unittest.main()
