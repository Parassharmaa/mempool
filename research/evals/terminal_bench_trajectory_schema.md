# Terminal-Bench Trajectory Schema

This schema is for internal Terminal-Bench 2.1 pilot records. It is deliberately
separate from the BigCodeBench routing dataset because Terminal-Bench rows are
multi-turn agent trajectories, not single prompt-response worker outcomes.

## Content Policy

Do not persist task instructions, prompts, oracle solutions, verifier code, raw
stdout, raw stderr, or full terminal transcripts in mempool artifacts. Store
metadata and compact behavioral summaries only.

Forbidden fields include:

- `instruction`
- `instructions`
- `prompt`
- `description`
- `solution`
- `oracle`
- `tests`
- `test_script`
- `stdout`
- `stderr`
- `transcript`
- `full_transcript`

## Required Fields

- `benchmark_id`: must be `terminal-bench-2.1`
- `run_id`
- `task_id`
- `trial_id`
- `agent_id`
- `worker_id`
- `policy_id`
- `selected_workflow`
- `task_success`
- `verifier_passed`
- `latency_ms`
- `cost_usd`
- `terminal_actions`
- `file_edits`
- `tests_run`
- `worker_switches`
- `failure_mode`

## Nested Summaries

Each `terminal_actions` item must include:

- `index`
- `action_kind`
- `summary`
- `exit_code`

Each `file_edits` item must include:

- `path`
- `operation`

Each `tests_run` item must include:

- `command_summary`
- `passed`

## Validation

Validate JSONL trajectory files with:

```bash
PYTHONPATH=src python3 tools/validate_terminal_bench_trajectories.py <path>
```

The first Terminal-Bench pilot should use this schema for both fixed-worker
agent baselines and orchestrator-selected runs. Keep Terminal-Bench held out
from policy training until a later explicit decision allows sanitized
trajectory-derived features.

## Metadata Selection Bridge

Before selecting pilot tasks, extract metadata-only rows from a local
Terminal-Bench export or task directory:

```bash
PYTHONPATH=src python3 tools/extract_terminal_bench_metadata.py \
  --input <terminal-bench-export-or-task-dir> \
  --output research/evals/terminal_bench_2p1_metadata.json \
  --report research/evals/terminal_bench_2p1_metadata_report.json
```

Then select the pilot manifest:

```bash
PYTHONPATH=src python3 tools/select_terminal_bench_pilot.py \
  --metadata research/evals/terminal_bench_2p1_metadata.json \
  --limit 5 \
  --output research/evals/terminal_bench_2p1_pilot_manifest.json
```

The real metadata-only Terminal-Bench 2.1 extraction writes:

- `research/evals/terminal_bench_2p1_metadata.json`
- `research/evals/terminal_bench_2p1_metadata_report.json`
- `research/evals/terminal_bench_2p1_pilot_manifest.json`

The example files under `research/evals/terminal_bench_metadata_example*.json`
are synthetic command-path fixtures, not benchmark results.

## Harness Preflight

Use the timeout-bounded preflight wrapper before running worker comparisons:

```bash
PYTHONPATH=src python3 tools/run_terminal_bench_preflight.py \
  --task-path external_repos/terminal-bench-2-1/tasks/cancel-async-tasks \
  --job-name mempool-tb2p1-install-cancel-async-retry \
  --jobs-dir research/runs/<run-tag>/harbor_jobs \
  --output research/evals/terminal_bench_2p1_install_retry_summary.json \
  --install-only \
  --timeout-seconds 600
```

The wrapper suppresses Harbor stdout/stderr capture in mempool artifacts and
summarizes only structured `result.json`/`config.json` fields.

Check readiness before launching worker comparisons:

```bash
PYTHONPATH=src python3 tools/check_terminal_bench_readiness.py \
  --summary research/evals/terminal_bench_2p1_fix_git_install_preflight_summary.json \
  --output research/evals/terminal_bench_2p1_readiness.json
```

A pilot run is acceptable only after at least one safe preflight summary reports
`process_status: exited` and `harbor_summary.status: complete`.
