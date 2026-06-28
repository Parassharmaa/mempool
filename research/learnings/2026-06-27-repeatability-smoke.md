# 2026-06-27 Cloud Repeatability Smoke

## Question

Should routing labels be based on single samples, or do cloud worker outcomes
need repeated measurements before training the logits-head orchestrator?

## Setup

- Worker pool: `research/evals/ollama_cloud_fast_compare_pool.json`
- Task set: `research/evals/bigcodebench_hard_qwen_positive_tasks.json`
- Limit: first 2 mined positive tasks
- Repeat count: 2 samples per worker/task
- Outcomes: `research/evals/results/20260627-cloud-repeatability-smoke.jsonl`
- Summary: `research/evals/results/20260627-cloud-repeatability-smoke-summary.json`
- Aggregated routing dataset:
  `research/datasets/20260627-cloud-repeatability-smoke-routing.jsonl`
- Router report:
  `research/datasets/20260627-cloud-repeatability-smoke-router-report.json`

## Result

The run produced 16 outcomes, summarized into 8 worker-task records.

Overall:

- `ollama-cloud-qwen3-coder-480b`: 4/4, 2831.8 ms mean latency
- `ollama-cloud-kimi-k2.7-code`: 4/4, 12347.0 ms mean latency
- `ollama-cloud-deepseek-v4-pro`: 3/4, 19483.2 ms mean latency
- `ollama-cloud-glm-5.2`: 2/4, 13897.2 ms mean latency

Per task:

- `BigCodeBench/13`: Qwen and Kimi were 2/2, DeepSeek V4 Pro was 1/2, GLM was
  0/2.
- `BigCodeBench/19`: all four workers were 2/2, making this mostly a latency
  and cost routing task.

## Harness Change

`tools/run_real_smoke_benchmark.py` now supports `--repeat-count` and writes a
`sample_index` for every record and JSONL outcome. Resume keys include
`sample_index`, so repeated probes can be resumed without clobbering earlier
samples.

`tools/summarize_repeated_outcomes.py` summarizes repeated outcomes into
worker-task pass rates, mean latency, sample pass sequences, and failure-mode
counts.

`tools/build_repeated_routing_dataset.py` converts repeated outcomes into a
schema-compatible routing dataset. It uses empirical pass rate as `score`,
mean latency as the route latency, and softmax targets over
pass-rate-minus-latency-penalty rewards.

## Decision

Trainable routing targets should aggregate repeated samples into empirical pass
rates and latency distributions. Single-sample labels are useful for smoke
testing, but they are too brittle for the logits-head orchestrator. The next
dataset step is to run this repeated-sample converter on a larger mined-positive
subset, then train the first logits-head prototype against its soft targets.
