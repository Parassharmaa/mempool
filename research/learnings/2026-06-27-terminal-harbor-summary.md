# Terminal-Bench Harbor Summary

## What Changed

Added a metadata-safe Harbor job summarizer for Terminal-Bench preflight and
pilot runs.

Artifacts:

- `src/mempool/terminal_bench.py`
- `tools/summarize_terminal_bench_harbor_job.py`
- `research/evals/terminal_bench_2p1_oracle_smoke_summary.json`
- `research/evals/terminal_bench_2p1_install_preflight_summary.json`

## Learning

The interrupted one-task oracle smoke was not just stale after interruption.
Harbor eventually wrote a structured result with completed, errored, and
cancelled counters all nonzero for the same single trial. The install-only
preflight produced the same interrupted shape. That state should not be
interpreted as a successful oracle sanity check or a successful environment
setup check.

The summarizer reads only `result.json` and `config.json`, and records
`raw_log_policy: not_read`. This gives us reproducible harness diagnostics
without copying Terminal-Bench instructions, verifier logs, transcripts, or raw
task output into mempool artifacts.

## Decision

All future Terminal-Bench pilot runs should emit a safe Harbor summary before
any trajectory conversion or model comparison. Raw Harbor logs stay under
ignored `research/runs/` paths.

## Next Step

Retry a one-task oracle smoke with a narrower Harbor/debug configuration, then
compare the safe summary against this interrupted baseline.
