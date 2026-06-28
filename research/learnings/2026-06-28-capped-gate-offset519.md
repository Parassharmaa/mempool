# Capped Gate Offset519

Run tag: `20260628-capped-gate-offset519`

## Question

Does the capped Qwen-first specialist-solvability gate continue to avoid
low-value specialist spend on a fresh, healthier eligible slice?

## Fresh Eligible Slice

Scanned `bigcode/bigcodebench` normal tasks from offset `519`.

Result:

- scanned rows: `77`
- eligible tasks: `16`
- next offset: `596`

Eligible tasks:

- `BigCodeBench-539`
- `BigCodeBench-542`
- `BigCodeBench-543`
- `BigCodeBench-544`
- `BigCodeBench-545`
- `BigCodeBench-547`
- `BigCodeBench-548`
- `BigCodeBench-563`
- `BigCodeBench-565`
- `BigCodeBench-566`
- `BigCodeBench-569`
- `BigCodeBench-570`
- `BigCodeBench-577`
- `BigCodeBench-592`
- `BigCodeBench-594`
- `BigCodeBench-595`

The rejected canonical rows were still mostly local dependency gaps:
`pandas`, `numpy`, `matplotlib`, `sklearn`, `pytz`, `rsa`, and related packages.

## Candidate Selection

The Qwen-fail contrast selector used:

- active learned policy registry
- rank-1 specialist filter
- prior Qwen-fail and Qwen-fast anchor ids from the offset236 boundary dataset
- exclusions for prior routing datasets and recent outcome screens

It selected four candidates:

- `BigCodeBench-594`
- `BigCodeBench-565`
- `BigCodeBench-595`
- `BigCodeBench-548`

The strongest candidate was `BigCodeBench-594`, a filesystem CSV/datetime task
with high Qwen-fail similarity:

- Qwen-fail similarity: `14.3378`
- Qwen-anchor similarity: `8.3277`
- active router top worker: Kimi

## Qwen-First Screen

Qwen outcomes:

- `BigCodeBench-594`: failed, `10814 ms`
- `BigCodeBench-565`: failed, `2553 ms`
- `BigCodeBench-595`: passed, `2472 ms`
- `BigCodeBench-548`: failed, `2090 ms`

Qwen-first removed the Qwen-fast row `595` from specialist consideration.

## Capped Solvability Gate

The capped gate used:

- `--min-gate-score 0`
- `--max-universal-fail-similarity 4`

Prior outcome memory included:

- specialist-positive top-four outcomes
- prior universal specialist failures
- the previous prospective universal failure from `BigCodeBench-365`

Gate result:

- rejected task count: `3`
- selected task ids: `[]`
- rejected by score: `2`
- rejected by universal-failure cap: `1`

Scored rejected rows:

- `BigCodeBench-594`: score `6.1962`, universal-failure similarity `5.7133`
- `BigCodeBench-565`: score `-2.7479`, universal-failure similarity `2.9413`
- `BigCodeBench-548`: score `-9.2139`, universal-failure similarity `5.9941`

The important row is `BigCodeBench-594`: it had a positive score and strong
Qwen-fail similarity, but the hard universal-failure cap blocked it before
specialist calls were spent.

## Interpretation

This run validates the tightened acquisition rule on a fresh slice.

Without the cap, `BigCodeBench-594` would have looked tempting because it was:

- Qwen-fail-like
- Kimi-top under the active router
- positive under the soft solvability score

The cap rejected it because it was also too close to known universal specialist
failures. That is the intended behavior after the previous prospective miss.

## Decision

Keep the capped gate as the default. Do not run specialists when the capped gate
selects no tasks.

No routing rows should be merged from this run. The useful artifacts are the
Qwen-first outcomes and the auditable skip decision.

## Next Step

Continue from offset `596`. The capped gate is now behaving well enough that the
next improvement should be acquisition breadth, not looser thresholds:

- scan the next eligible slice
- select fresh Qwen-resistant rank-1 specialist candidates
- run Qwen first
- only run specialists if `--min-gate-score 0` and
  `--max-universal-fail-similarity 4` both pass
