# 2026-06-27 Four-Task Logits Router

## Question

Can the logits-head prototype train on a larger repeated-sample routing dataset
covering all mined Qwen-positive tasks?

## Setup

- First repeated outcomes:
  `research/evals/results/20260627-cloud-repeatability-smoke.jsonl`
- Remaining repeated outcomes:
  `research/evals/results/20260627-cloud-repeatability-remaining.jsonl`
- Merged outcomes:
  `research/evals/results/20260627-cloud-repeatability-4task.jsonl`
- Summary:
  `research/evals/results/20260627-cloud-repeatability-4task-summary.json`
- Dataset:
  `research/datasets/20260627-cloud-repeatability-4task-routing.jsonl`
- Model:
  `research/models/20260627-repeatability-4task-logits-router.json`
- Report:
  `research/datasets/20260627-repeatability-4task-logits-router-report.json`

## Result

The merged repeatability set has 32 outcomes: 4 tasks, 4 workers, 2 samples per
worker/task.

Aggregate pass rates:

- `ollama-cloud-qwen3-coder-480b`: 8/8, mean latency 2587.88 ms
- `ollama-cloud-kimi-k2.7-code`: 5/8, mean latency 9773.62 ms
- `ollama-cloud-deepseek-v4-pro`: 3/8, mean latency 12706.75 ms
- `ollama-cloud-glm-5.2`: 2/8, mean latency 12993.5 ms

The four-task logits router uses 45 prompt features and 4 worker logits. Training
on the empirical soft targets produced:

- Initial mean KL: 0.5609365496181639
- Final training mean KL: 0.00010436958331083471
- Evaluation mean KL: 0.00010328067379894974
- Target accuracy: 4/4
- pass@1: 4/4

## Interpretation

The trainable policy path now works over all currently mined positive tasks.
However, the hard target is still Qwen Coder for every task, so this is mainly a
soft-target fitting result. It is not yet evidence that the router can choose
between different best workers.

The next data objective is to mine or construct repeated benchmark slices where
non-Qwen workers are empirical winners under reliability, latency, cost, or
specialized task-family rewards.
