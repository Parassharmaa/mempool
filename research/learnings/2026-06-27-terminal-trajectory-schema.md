# Terminal-Bench Trajectory Schema

## What Changed

Added a metadata-safe trajectory schema and validator for future Terminal-Bench
2.1 pilot runs.

Artifacts:

- `src/mempool/terminal_bench.py`
- `tools/validate_terminal_bench_trajectories.py`
- `research/evals/terminal_bench_trajectory_schema.md`
- `research/evals/terminal_bench_2p1_pilot_plan.json`

## Learning

Terminal-Bench should enter mempool as an agentic trajectory harness, not as a
second single-step routing dataset. Its records need a stricter content policy
than BigCodeBench routing rows because raw terminal output, task instructions,
test scripts, and transcripts can leak benchmark content into training data.

The first safe bridge is summary-only trajectory JSONL:

- task and trial metadata
- selected worker and policy
- compact terminal action summaries
- file-edit and verifier summaries
- success, latency, cost, worker-switch, and failure-mode fields

## Decision

Keep Terminal-Bench held out from policy training for now. Use the new validator
as the acceptance gate for pilot run artifacts, then decide later whether
sanitized state/history features are suitable for orchestrator training.

## Next Step

Materialize a 3-5 task metadata-only Terminal-Bench subset and run an oracle or
harness sanity check before spending worker calls.
