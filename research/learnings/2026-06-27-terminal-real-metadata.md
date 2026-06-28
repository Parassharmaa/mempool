# Terminal-Bench Real Metadata Extraction

## What Changed

Cloned the official Terminal-Bench 2.1 dataset repository into
`external_repos/terminal-bench-2-1` as a reference checkout, then extracted only
metadata-safe task rows into mempool artifacts.

Artifacts:

- `research/evals/terminal_bench_2p1_metadata.json`
- `research/evals/terminal_bench_2p1_metadata_report.json`
- `research/evals/terminal_bench_2p1_pilot_manifest.json`
- `research/evals/terminal_bench_2p1_harness_preflight.json`
- `research/evals/terminal_bench_2p1_oracle_smoke_summary.json`
- `src/mempool/terminal_bench.py`

## Learning

The real Terminal-Bench 2.1 repository contains 89 tasks. Each task root has a
`task.toml` file with a safe `[metadata]` section containing category,
difficulty, and tags. The task directories also contain environment, solution,
test, README, and instruction files; those must remain outside mempool training
artifacts.

The extractor initially picked up environment subdirectories because they
contain Dockerfiles. Tightening the scanner to skip `environment`, `solution`,
and `tests` directories produced the expected 89 metadata rows.

The first metadata-only 5-task pilot manifest selects:

- `terminal-bench/cancel-async-tasks`
- `terminal-bench/configure-git-webserver`
- `terminal-bench/financial-document-processor`
- `terminal-bench/fix-code-vulnerability`
- `terminal-bench/train-fasttext`

## Decision

Keep these rows as pilot-selection metadata only. Do not use Terminal-Bench task
instructions, verifier code, oracle solutions, raw logs, or transcripts for
training.

Harbor and Docker are available locally, but the first one-task oracle smoke on
`terminal-bench/cancel-async-tasks` did not complete cleanly. It was interrupted
after about 211 seconds. A safe Harbor summary now records the final interrupted
state as `errored`, with completed, errored, and cancelled counters all nonzero.
Treat this as a harness preflight issue, not a benchmark result.

## Next Step

Debug the Harbor oracle smoke path on one selected task before running fixed
worker or orchestrator comparisons. Keep BigCodeBench data acquisition as the
active scoring path until Terminal-Bench produces a reproducible oracle result.
