# Router Miss Repeat Live Run

## Question

Do the sparse leave-one-out miss seeds remain stable when repeated across the
top4 cloud worker pool?

## Result

Ran the ready repeat-stabilization manifest:

- manifest: `research/evals/20260628-router-miss-repeat4-top4-manifest.json`
- outcomes: `research/evals/results/20260628-router-miss-repeat4-top4.jsonl`
- summary: `research/evals/results/20260628-router-miss-repeat4-top4-summary.json`
- routing rows: `research/datasets/20260628-router-miss-repeat4-top4-routing.jsonl`

The run produced 32 worker calls over 4 tasks, 4 workers, and 2 repeats.

Observed routing signal:

- `BigCodeBench-5`: stable broad-pass row; GLM is the fastest empirical target.
- `BigCodeBench-368`: unstable DeepSeek signal, 1/2 pass rate.
- `BigCodeBench-763`: all-fail in this repeat run.
- `BigCodeBench-963`: all-fail in this repeat run.

The merge-readiness filter kept only one row:

- `research/datasets/20260628-router-miss-repeat4-top4-merge-ready-routing.jsonl`

and dropped three rows:

- `368`: unstable target
- `763`: all-fail
- `963`: all-fail

## Refresh Gate

Merged the one stable row with the active 23-task dataset:

- `research/datasets/20260628-router-miss-repeat-24task-routing.jsonl`

The default tolerance gate selected temperature `0.05`, because latency regret
improved from 501.1 ms to 373.3 ms while target accuracy remained within the old
allowed drop.

The stricter accuracy-preserving gate quarantined the same candidate:

- baseline target accuracy: 0.783
- candidate target accuracy: 0.708
- baseline solvable pass@1: 0.800
- candidate solvable pass@1: 0.762
- baseline latency regret: 501.1 ms
- candidate latency regret: 373.3 ms

## Decision

Do not promote this candidate to the active policy. Keep the data and artifacts
as repeatability evidence.

The main learning is that sparse miss seeds are noisy: repeating them converted
two apparent router-boundary examples into all-fail evidence and one into an
unstable target. The next acquisition step should favor stable broad-pass
latency rows and repeated specialist wins before expanding model capacity.
