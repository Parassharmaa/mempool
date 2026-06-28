import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.run_terminal_bench_preflight import (
    build_harbor_preflight_command,
    run_harbor_preflight,
)
from mempool.terminal_bench import (
    evaluate_terminal_bench_readiness,
    refresh_terminal_bench_preflight_summary,
)


class TerminalBenchPreflightRunnerTest(unittest.TestCase):
    def test_builds_install_only_command(self) -> None:
        command = build_harbor_preflight_command(
            Path("tasks/example"),
            "job-name",
            Path("jobs"),
            install_only=True,
        )

        self.assertEqual(command[:3], ["uvx", "harbor", "run"])
        self.assertIn("--install-only", command)
        self.assertIn("--path", command)
        self.assertIn("tasks/example", command)
        self.assertIn("--jobs-dir", command)
        self.assertIn("jobs", command)

    def test_builds_model_agent_kwargs_and_host_allowlist(self) -> None:
        command = build_harbor_preflight_command(
            Path("tasks/example"),
            "job-name",
            Path("jobs"),
            agent="terminus-2",
            model="openai/qwen3-coder-next",
            agent_kwargs=("api_base=https://ollama.com/v1", "max_turns=5"),
            allow_agent_hosts=("ollama.com",),
        )

        self.assertIn("--model", command)
        self.assertIn("openai/qwen3-coder-next", command)
        self.assertEqual(command.count("--agent-kwarg"), 2)
        self.assertIn("api_base=https://ollama.com/v1", command)
        self.assertIn("max_turns=5", command)
        self.assertIn("--allow-agent-host", command)
        self.assertIn("ollama.com", command)

    def test_writes_safe_summary_after_exit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "jobs" / "job"
            job_dir.mkdir(parents=True)
            (job_dir / "config.json").write_text(
                '{"environment": {"type": "docker"}, "agent": {"type": "oracle"}}',
                encoding="utf-8",
            )
            (job_dir / "result.json").write_text(
                """{
  "id": "job",
  "finished_at": "2026-06-27T00:00:01Z",
  "n_total_trials": 1,
  "stats": {
    "n_completed_trials": 1,
    "n_errored_trials": 0,
    "n_cancelled_trials": 0,
    "n_running_trials": 0,
    "n_pending_trials": 0
  }
}""",
                encoding="utf-8",
            )

            def fake_runner(*args, **kwargs):
                return subprocess.CompletedProcess(args[0], 0)

            output = root / "summary.json"
            payload = run_harbor_preflight(
                ["uvx", "harbor", "run"],
                job_dir,
                output,
                timeout_seconds=1,
                runner=fake_runner,
            )

        self.assertEqual(payload["process_status"], "exited")
        self.assertEqual(payload["returncode"], 0)
        self.assertEqual(payload["harbor_summary"]["status"], "complete")
        self.assertEqual(payload["raw_log_policy"], "not_read")

    def test_records_timeout_without_result(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def timeout_runner(*args, **kwargs):
                raise subprocess.TimeoutExpired(args[0], timeout=1)

            payload = run_harbor_preflight(
                ["uvx", "harbor", "run"],
                root / "missing-job",
                root / "summary.json",
                timeout_seconds=1,
                runner=timeout_runner,
            )

        self.assertEqual(payload["process_status"], "timeout")
        self.assertIsNone(payload["returncode"])
        self.assertIsNone(payload["harbor_summary"])

    def test_readiness_gate_accepts_complete_summary(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                """{
  "process_status": "exited",
  "harbor_summary": {"status": "complete"}
}""",
                encoding="utf-8",
            )

            result = evaluate_terminal_bench_readiness([path])

        self.assertTrue(result["ready"])
        self.assertTrue(result["checks"][0]["ready"])

    def test_readiness_gate_rejects_timeout_summary(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                """{
  "process_status": "timeout",
  "harbor_summary": {"status": "running_or_stale"}
}""",
                encoding="utf-8",
            )

            result = evaluate_terminal_bench_readiness([path])

        self.assertFalse(result["ready"])
        self.assertEqual(
            result["checks"][0]["reasons"],
            ["process_status=timeout", "harbor_status=running_or_stale"],
        )

    def test_refreshes_legacy_harbor_summary_without_logs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "jobs" / "job"
            job_dir.mkdir(parents=True)
            (job_dir / "config.json").write_text(
                '{"environment": {"type": "docker"}, "agent": {"type": "oracle"}}',
                encoding="utf-8",
            )
            (job_dir / "result.json").write_text(
                """{
  "id": "job",
  "finished_at": "2026-06-27T00:00:01Z",
  "n_total_trials": 1,
  "stats": {
    "n_completed_trials": 1,
    "n_errored_trials": 0,
    "n_cancelled_trials": 0,
    "n_running_trials": 0,
    "n_pending_trials": 0
  }
}""",
                encoding="utf-8",
            )
            legacy_summary = root / "legacy-summary.json"
            legacy_summary.write_text(
                json.dumps(
                    {
                        "job_dir": str(job_dir),
                        "status": "running_or_stale",
                        "raw_log_policy": "not_read",
                    }
                ),
                encoding="utf-8",
            )

            refreshed = refresh_terminal_bench_preflight_summary(legacy_summary)

        self.assertIsNone(refreshed["process_status"])
        self.assertEqual(refreshed["harbor_summary"]["status"], "complete")
        self.assertEqual(refreshed["raw_log_policy"], "not_read")
        self.assertEqual(
            refreshed["refresh_policy"],
            "metadata_only_result_json_and_config_json",
        )


if __name__ == "__main__":
    unittest.main()
