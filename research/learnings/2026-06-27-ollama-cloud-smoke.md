# 2026-06-27 Ollama Cloud Smoke

## Question

Can the worker harness compare current cloud-scale workers through the Ollama
OpenAI-compatible endpoint and turn provider behavior into router training data?

## Setup

- Worker pool: `research/evals/ollama_cloud_worker_pool.json`
- Task set: `research/evals/bigcodebench_hard_eligible_merged_tasks.json`
- Limit: 1 BigCodeBench-Hard eligible task
- Run: `research/evals/results/20260627-ollama-cloud-smoke-1.jsonl`
- Routing dataset: `research/datasets/20260627-ollama-cloud-smoke-1-routing.jsonl`
- Router report: `research/datasets/20260627-ollama-cloud-smoke-1-router-report.json`

## Result

On `bigcodebench-hard-BigCodeBench-13`:

- `ollama-cloud-qwen3-coder-480b` passed in 3077 ms.
- `ollama-cloud-deepseek-v4-pro` passed in 11632 ms.
- `ollama-cloud-glm-5.2` failed evaluator tests in 9562 ms.
- `ollama-cloud-kimi-k2.7-code` failed evaluator tests in 9922 ms.
- `ollama-cloud-deepseek-v3.2` hit the 90 second request timeout and was
  recorded as `request_timeout`.

The router baseline therefore selects `ollama-cloud-qwen3-coder-480b` as both
the strongest and fastest worker on this tiny smoke sample. This is not yet a
benchmark claim; it is a plumbing and first-signal result.

## Harness Change

`tools/run_real_smoke_benchmark.py` now records provider exceptions as benchmark
outcomes instead of crashing the entire run. Request timeouts become
`failure_mode=request_timeout`; other provider exceptions become
`failure_mode=request_error`. This matters for orchestration because latency
and availability are part of the route quality signal.

## Implication

Use BigCodeBench-Hard as the immediate supervised label source for the logits
router, then add Terminal-Bench as the agentic harness once multi-worker runs can
survive slow or unavailable providers. Terminal-Bench should measure stateful
terminal workflows, tool use, verification behavior, and worker switching, not
replace the cleaner single-task routing labels.
