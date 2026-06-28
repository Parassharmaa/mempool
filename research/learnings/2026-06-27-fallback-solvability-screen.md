# Fallback Solvability Screen

## Question

Can a cheap one-sample screen find solvable fallback-opportunity tasks before
spending repeated samples?

## Change

Added `tools/select_solvability_screen_batch.py` to select a one-sample screen
batch from the fallback-opportunity ranked candidates while excluding known hard
negatives.

Generated:

- `research/evals/bigcodebench_hard_fallback_solvability_screen4_tasks.json`
- `research/evals/bigcodebench_hard_fallback_solvability_screen4_report.json`

Then ran the selected 4 tasks once across the top-4 cloud worker pool and wrote:

- `research/evals/results/20260627-fallback-solvability-screen4.json`
- `research/evals/results/20260627-fallback-solvability-screen4.jsonl`
- `research/evals/results/20260627-fallback-solvability-screen4-summary.json`
- `research/datasets/20260627-fallback-solvability-screen4-routing.jsonl`
- `research/datasets/20260627-fallback-solvability-screen4-fallback-training.jsonl`
- `research/datasets/20260627-fallback-solvability-screen4-fallback-training-report.json`

## Result

The screen selected:

- `BigCodeBench/509`
- `BigCodeBench/857`
- `BigCodeBench/15`
- `BigCodeBench/346`

All four were universal failures across the top-4 cloud pool with one sample per
worker. The fallback-specific report contains:

- task count: 4
- fallback opportunities: 4
- useful fallbacks: 0
- fallback hurts: 4
- solvable tasks: 0

Combined with the previous top-2 repeated run, the fallback acquisition path has
now found 6 fallback opportunities and 0 useful fallback positives from
uncertainty-first selection.

## Learning

The cheap screen helped avoid repeated sampling, but it also confirmed that
active-router uncertainty is not enough. The fallback data pipeline now correctly
records negative action labels, but the acquisition strategy needs a stronger
solvability prior.

The next selector should start from known-pass neighborhoods rather than from
low-margin uncertainty alone:

- choose tasks similar to rows where at least one worker already passed, or
- mine with the strongest worker first and only fallback-screen tasks with a
  positive top-worker or specialist signal, or
- use canonical-pass plus low environment risk as a prerequisite before cloud
  fallback screening.

## Next Step

Build a positive-neighborhood fallback selector seeded from existing successful
tasks, then screen a small batch where at least one worker is more likely to
pass. Keep the current six rows as negative fallback-action training examples,
but do not merge them into router promotion datasets as fastest-failure wins.
