# Capped Gate Offset596

Run tag: `20260628-capped-gate-offset596`

## Question

Does the capped Qwen-first specialist-solvability gate continue to avoid
specialist spend on the fresh slice after offset `596`?

## Fresh Eligible Slice

Scanned `bigcode/bigcodebench` normal tasks from offset `596`.

Result:

- scanned rows: `80`
- eligible tasks: `13`
- next offset: `676`

Eligible tasks:

- `BigCodeBench-604`
- `BigCodeBench-642`
- `BigCodeBench-644`
- `BigCodeBench-665`
- `BigCodeBench-666`
- `BigCodeBench-667`
- `BigCodeBench-668`
- `BigCodeBench-669`
- `BigCodeBench-670`
- `BigCodeBench-671`
- `BigCodeBench-672`
- `BigCodeBench-673`
- `BigCodeBench-675`

The rejected canonical rows were mostly dependency gaps such as `pandas`,
`numpy`, `matplotlib`, `seaborn`, `pytz`, and `nltk`.

## Candidate Selection

The strict rank-1 specialist selector produced no candidates on this slice.

Relaxing only the specialist-rank filter produced four Qwen-top candidates:

- `BigCodeBench-671`
- `BigCodeBench-673`
- `BigCodeBench-672`
- `BigCodeBench-675`

All four had Qwen as the active router top worker with near-total confidence,
while the strongest specialist was only rank 2.

## Qwen-First Screen

Qwen outcomes:

- `BigCodeBench-671`: passed, `14845 ms`
- `BigCodeBench-673`: passed, `5793 ms`
- `BigCodeBench-672`: passed, `5445 ms`
- `BigCodeBench-675`: passed, `7817 ms`

Qwen-first eliminated all four candidates from specialist consideration.

## Capped Solvability Gate

The capped gate used:

- `--max-universal-fail-similarity 0.1`
- prior specialist-positive and universal-failure outcome memory

Gate result:

- rejected task count: `0`
- selected task ids: `[]`
- specialist calls run: `0`

## Harness Finding

This run exposed a real harness-safety issue. One BigCodeBench candidate test
removed `./src` during teardown. The smoke evaluator had been launching
candidate subprocesses from the repository root, so the benchmark test deleted
the local source tree.

The evaluator now runs candidate tests with `cwd` set to the temporary candidate
directory. The focused regression check verified that `src/mempool` survives the
same smoke and BigCodeBench tests.

## Interpretation

The capped gate is still conservative, but this slice did not produce the target
Qwen-resistant specialist opportunity. Once rank-1 specialist selection was
relaxed, the selector mostly found filesystem tasks that Qwen solved cleanly.

This is useful negative evidence: the next acquisition step should search for
stronger non-Qwen priors or harder categories, not spend specialists on
Qwen-top filesystem rows.

## Decision

Keep the capped gate and the Qwen-first screen. Do not merge routing rows from
this run because no specialist comparison was needed.

Also keep the evaluator isolation fix as harness protection before any further
live benchmark runs.
