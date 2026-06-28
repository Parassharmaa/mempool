# 2026-06-27 Qwen Positive Mining

## Question

Can we cheaply mine more positive BigCodeBench-Hard tasks before spending full
multi-worker cloud comparisons?

## Setup

Positive mining:

- Worker pool: `research/evals/ollama_cloud_qwen_mining_pool.json`
- Task set: `research/evals/bigcodebench_hard_eligible_merged_tasks.json`
- Outcomes: `research/evals/results/20260627-qwen-positive-mining-16.jsonl`
- Selected tasks: `research/evals/bigcodebench_hard_qwen_positive_tasks.json`

Full comparison on mined positives:

- Worker pool: `research/evals/ollama_cloud_fast_compare_pool.json`
- Outcomes: `research/evals/results/20260627-cloud-positive-compare-4.jsonl`
- Routing dataset:
  `research/datasets/20260627-cloud-positive-compare-4-routing.jsonl`
- Router report:
  `research/datasets/20260627-cloud-positive-compare-4-router-report.json`

## Result

`ollama-cloud-qwen3-coder-480b` solved 4/16 eligible tasks during mining:

- `bigcodebench-hard-BigCodeBench-13`
- `bigcodebench-hard-BigCodeBench-19`
- `bigcodebench-hard-BigCodeBench-454`
- `bigcodebench-hard-BigCodeBench-777`

The faster cloud comparison pool, excluding the repeatedly slow
`deepseek-v3.2`, produced:

- `ollama-cloud-qwen3-coder-480b`: 4/4 solved, 2662.8 ms mean latency
- `ollama-cloud-glm-5.2`: 3/4 solved, 10217.2 ms mean latency
- `ollama-cloud-kimi-k2.7-code`: 3/4 solved, 13753.8 ms mean latency
- `ollama-cloud-deepseek-v4-pro`: 2/4 solved, 13719.8 ms mean latency

The hard target still selects Qwen Coder for every task because it is both
fastest and most reliable on this subset. The soft target distribution is more
useful: tasks 13, 19, and 777 have multiple passing workers, while task 454 is
almost Qwen-only.

## Decision

Positive mining is a useful stage before expensive multi-worker comparisons.
Keep using it to build a compact set of solvable, differentiating tasks. Do not
move to logits-head training until the mined subset contains enough examples
where non-Qwen workers are the best target under at least one objective, such as
latency-adjusted reward, cost-adjusted reward, or repeatability.
