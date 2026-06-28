# Capped Solvability Gate Next Slice

Run tag: `20260628-capped-solvability-gate-next`

## Question

After adding the universal-failure similarity cap, does the Qwen-first
acquisition loop avoid specialist spend on a fresh slice when the rejected task
is not sufficiently specialist-solvable?

## Fresh Eligible Slice

Scanned `bigcode/bigcodebench` normal tasks from offset `439`.

Result:

- scanned rows: `80`
- eligible tasks: `7`
- next offset: `519`

Eligible tasks:

- `BigCodeBench-453`
- `BigCodeBench-454`
- `BigCodeBench-459`
- `BigCodeBench-466`
- `BigCodeBench-505`
- `BigCodeBench-508`
- `BigCodeBench-510`

The rejected canonical rows were again dominated by local dependency gaps:
`numpy`, `pandas`, `matplotlib`, `scipy`, `pytz`, `xlwt`, and similar packages.

## Candidate Selection

The Qwen-fail contrast selector used:

- active learned policy registry
- rank-1 specialist filter
- prior Qwen-fail and Qwen-fast anchor ids from the offset236 boundary dataset
- exclusions for prior routing datasets and recent Qwen/specialist outcome files

It selected one candidate:

- `BigCodeBench-510`

Profile:

- categories: filesystem
- libraries: `difflib`, `gzip`
- active router top worker: Kimi
- second worker: Qwen
- first-second margin: `0.0062`
- Qwen-fail similarity: `2.4946`
- Qwen-anchor similarity: `2.4955`

This was a weak but plausible boundary case: specialist top, Qwen close second,
and low margin.

## Qwen-First Screen

Qwen was run first on `BigCodeBench-510`.

Result:

- Qwen failed
- latency: `90122 ms`

Without a second gate, this would normally trigger three specialist calls.

## Capped Solvability Gate

The capped gate was run with:

- `--min-gate-score 0`
- `--max-universal-fail-similarity 4`

Prior outcome memory included:

- specialist-positive top-four outcomes
- prior universal specialist failures
- the previous prospective universal failure from `BigCodeBench-365`

Gate result:

- rejected task count: `1`
- selected task ids: `[]`
- rejected by score: `1`
- rejected by universal-failure cap: `0`

Scored rejection:

- `BigCodeBench-510`
- positive similarity: `2.4999`
- universal-failure similarity: `2.4995`
- score: `-1.4075`

The row was rejected by the score gate before specialist calls were spent.

## Interpretation

This is the first prospective run where the tightened Qwen-first plus
solvability gate avoided specialist spend after a Qwen failure.

The model-call sequence was:

1. one Qwen call
2. no specialist calls

That is exactly the acquisition-control behavior we want when the task is not
close enough to measured specialist-positive memory.

No new positive routing labels were produced, but the negative Qwen outcome and
the skipped-specialist decision are useful policy evidence.

## Decision

Keep the capped gate as the default specialist acquisition guard.

Do not merge routing rows from this run. There is no multi-worker positive
comparison row.

## Next Step

Scan the next eligible slice from offset `519`. Continue using:

- Qwen-first screening
- `--min-gate-score 0`
- `--max-universal-fail-similarity 4`

Only run specialists when the capped gate selects at least one task.
