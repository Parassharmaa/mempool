# Post-Refresh Held-Out Hard Diagnostic

## Question

After promoting the 23-task heldout-hard router, does the active policy
generalize to another fresh hard batch from the expanded top-four BigCodeBench
pool?

## Setup

Selected three fresh tasks after excluding all prior routing datasets and
outcome JSONL task IDs:

- `BigCodeBench/124`
- `BigCodeBench/526`
- `BigCodeBench/952`

Each task was run twice per worker across GLM 5.2, DeepSeek V4 Pro, Kimi K2.7
Code, and Qwen3 Coder 480B.

## Result

Only `BigCodeBench/526` was solvable:

- GLM 5.2: 2/2, mean latency 5032.5 ms
- DeepSeek V4 Pro: 1/2, mean latency 4985.5 ms
- Kimi K2.7 Code: 0/2
- Qwen3 Coder 480B: 0/2

The active 23-task logits router predicted Qwen for all three fresh rows. This
gave:

- pass@1: 0.0
- solvable-row pass@1: 0.0 over 1 solvable row
- target accuracy: 0.6667
- solvable target accuracy: 0.0

The strongest-worker baseline for this slice is GLM and solves 1/3.

## Interpretation

The previous refresh fixed a DeepSeek-stable held-out row but did not solve the
broader non-Qwen default problem. Fresh GLM-stable data/filesystem rows remain
underrepresented, and all-fail rows still make Qwen attractive on latency.

Do not immediately promote another one-row refresh from this result. The pattern
now points to a model-capacity or decision-structure issue: the linear feature
router needs either more GLM-positive evidence, a solvability-aware second
stage, or a verifier/abstain head that can avoid fastest-failure routing.

## Next Step

Use this as a regression slice. The next implementation should add a
solvability-aware report/gate or verifier/abstain target before another policy
promotion, then compare against this fresh GLM slice.

The regression slices are now machine-readable in
`research/evals/router_regression_slices.json` and can be checked with
`tools/evaluate_router_regression_slices.py`.
