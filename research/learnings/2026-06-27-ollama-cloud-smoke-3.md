# 2026-06-27 Ollama Cloud Three-Task Smoke

## Question

Does a slightly larger cloud worker run produce enough BigCodeBench-Hard signal
to train beyond the trivial strongest-worker baseline?

## Setup

- Worker pool: `research/evals/ollama_cloud_worker_pool.json`
- Task set: `research/evals/bigcodebench_hard_eligible_merged_tasks.json`
- Limit: 3 BigCodeBench-Hard eligible tasks
- Outcomes: `research/evals/results/20260627-ollama-cloud-smoke-3.jsonl`
- Routing dataset: `research/datasets/20260627-ollama-cloud-smoke-3-routing.jsonl`
- Router report: `research/datasets/20260627-ollama-cloud-smoke-3-router-report.json`

## Result

The run produced 15 worker outcomes grouped into 3 routing records.

- `ollama-cloud-qwen3-coder-480b`: 1/3 solved, 2799.7 ms mean latency
- `ollama-cloud-glm-5.2`: 0/3 solved, 6598.0 ms mean latency
- `ollama-cloud-deepseek-v4-pro`: 0/3 solved, 11699.0 ms mean latency
- `ollama-cloud-deepseek-v3.2`: 0/3 solved, 77767.0 ms mean latency
- `ollama-cloud-kimi-k2.7-code`: 0/3 solved, 11321.3 ms mean latency

All router baselines select `ollama-cloud-qwen3-coder-480b`, with pass@1 of
1/3. Leave-one-out checks are now technically available, but they are not yet
meaningful because every positive label points to the same worker.

## Repeatability Note

`ollama-cloud-deepseek-v4-pro` passed `BigCodeBench/13` in the one-task smoke
but failed the same task in the clean three-task run. The client already sends
`temperature: 0`, so repeatability still needs to be measured directly. The
OpenAI-compatible adapter now supports config-level `chat_options` so future
runs can pass provider-supported controls such as `seed` when available.

## Decision

Do not train the neural/logits-head orchestrator yet. Continue expanding
positive BigCodeBench outcomes first, either by scanning more standard-library
eligible tasks or adding an isolated dependency profile for benchmark tasks that
need data-science or web packages.
