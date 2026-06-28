# 2026-06-27 Kimi Winner Mining

## Question

Can we find benchmark tasks where a non-Qwen cloud worker is the empirical
winner, so the logits router learns real routing rather than a Qwen default?

## Setup

Kimi mining:

- Qwen-negative tasks: `research/evals/bigcodebench_hard_qwen_negative_tasks.json`
- Kimi mining pool: `research/evals/ollama_cloud_kimi_mining_pool.json`
- Outcomes: `research/evals/results/20260627-kimi-qwen-negative-mining-12.jsonl`
- Selected tasks:
  `research/evals/bigcodebench_hard_kimi_positive_qwen_negative_tasks.json`

Repeated comparison:

- Comparison outcomes:
  `research/evals/results/20260627-kimi-candidate-repeat-compare.jsonl`
- Summary:
  `research/evals/results/20260627-kimi-candidate-repeat-compare-summary.json`
- Routing dataset:
  `research/datasets/20260627-kimi-candidate-repeat-compare-routing.jsonl`

Mixed-winner training:

- Merged dataset: `research/datasets/20260627-mixed-winner-6task-routing.jsonl`
- Model: `research/models/20260627-mixed-winner-6task-logits-router.json`
- Report: `research/datasets/20260627-mixed-winner-6task-logits-router-report.json`

## Result

Kimi solved 2/12 tasks that Qwen failed in the earlier positive-mining pass:

- `bigcodebench-hard-BigCodeBench-310`
- `bigcodebench-hard-BigCodeBench-592`

The repeated comparison on those two tasks produced a clear non-Qwen winner
slice:

- `ollama-cloud-kimi-k2.7-code`: 4/4, mean latency 7491.5 ms
- `ollama-cloud-deepseek-v4-pro`: 1/4, mean latency 6674.25 ms
- `ollama-cloud-glm-5.2`: 1/4, mean latency 19047.75 ms
- `ollama-cloud-qwen3-coder-480b`: 0/4, mean latency 4434.75 ms

The merged six-task routing dataset now has mixed hard targets:

- Qwen Coder: 4 tasks
- Kimi K2.7 Code: 2 tasks

The mixed-winner logits router trained cleanly:

- Initial mean KL: 0.6680712727416127
- Final training mean KL: 0.0009673499261504121
- Evaluation mean KL: 0.0009576783661455036
- Target accuracy: 6/6
- pass@1: 6/6

## Decision

The data strategy works: mine failures from the current default worker, probe a
specialist, then repeat-compare candidates before adding them to the training
set. This gives the logits router actual worker-choice structure. Continue this
pattern across more specialists and task families before moving to a larger
language-model backbone.
