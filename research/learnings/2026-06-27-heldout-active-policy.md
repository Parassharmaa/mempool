# Held-Out Active Policy Diagnostic

Run tag: `20260627-heldout-active-policy`

## Question

Does the promoted active logits router generalize to a small held-out repeated
BigCodeBench slice that was not part of its eight-task training dataset?

## Setup

- Tasks: `BigCodeBench/906`, `BigCodeBench/928`
- Workers: GLM 5.2, DeepSeek V4 Pro, Kimi K2.7 Code, Qwen3 Coder 480B
- Samples: 2 per worker/task
- Outcome file:
  `research/evals/results/20260627-heldout-both-repeat-compare-env.jsonl`
- Held-out routing dataset:
  `research/datasets/20260627-heldout-both-repeat-routing.jsonl`

An initial run without the shell environment loaded produced only
`request_error` rows. The runner now loads repo `.env` values by default without
overwriting already-exported variables.

## Result

All four fast cloud workers passed both held-out tasks in both samples. The
empirical target was therefore latency-driven:

- Qwen3 Coder 480B: 4/4, mean latency 2372.25 ms
- DeepSeek V4 Pro: 4/4, mean latency 7690.25 ms
- GLM 5.2: 4/4, mean latency 8075.0 ms
- Kimi K2.7 Code: 4/4, mean latency 9050.75 ms

The active logits router selected Kimi for both tasks:

- pass@1: 2/2
- target accuracy: 0/2
- mean latency: 9050.5 ms
- mean KL: 0.8655

The family, nearest-neighbor, strongest-worker, fastest-worker, and oracle
baselines all selected Qwen on this two-task dataset.

## Interpretation

The active policy can still solve this held-out slice, but it misses the
latency target when capability ties. This is a useful failure: the current
feature space and training data are not strong enough to separate broad-pass
fast-Qwen tasks from Kimi-favored filesystem tasks.

## Next Step

Before moving to a larger backbone or Terminal-Bench, add more held-out
latency-tie and specialist-win examples, then train/evaluate a refresh that
reduces latency regret without losing pass@1.
