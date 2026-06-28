# Terminal-Bench Preflight Runner

## What Changed

Added a timeout-bounded wrapper for Harbor Terminal-Bench preflights.

Artifacts:

- `tools/run_terminal_bench_preflight.py`
- `tests/test_run_terminal_bench_preflight.py`
- `research/evals/terminal_bench_trajectory_schema.md`

## Learning

Manual Harbor retries are too easy to leave ambiguous. The previous oracle and
install-only attempts had to be interrupted by hand, then interpreted from
structured result files. The new wrapper makes that pattern reproducible:

- builds the Harbor command from explicit task/job parameters
- applies a process timeout
- suppresses Harbor stdout/stderr capture in mempool artifacts
- summarizes only structured Harbor result/config fields
- writes one safe JSON artifact for gate decisions

## Decision

Use this wrapper for the next Terminal-Bench harness diagnostic. Do not run
worker comparisons until a wrapper-produced safe summary reports
`status: complete`.

## Next Step

Retry the selected `cancel-async-tasks` install-only preflight with a longer
timeout, or try a simpler selected task, and compare the wrapper summary against
the interrupted ambiguous baselines.
