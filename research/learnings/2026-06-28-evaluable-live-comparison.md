# Evaluable Live Comparison

Run tag: `20260628-evaluable-live-comparison`

## What Changed

Ran the selected current-environment evaluable prompt set through the live
orchestrated prompt-set harness. The comparison used the trained task-level
orchestrator against a fixed `ollama-cloud-qwen3-coder-480b` baseline.

Artifacts:

- `research/evals/20260628-evaluable-live-comparison-prompts.json`
- `research/evals/20260628-evaluable-live-comparison.json`
- `research/evals/20260628-evaluable-live-comparison-outcomes.jsonl`
- `research/evals/20260628-evaluable-live-comparison-evaluation.jsonl`
- `research/evals/20260628-evaluable-live-comparison-evaluation-report.json`

## Result

All six execution rows were locally evaluable; no rows were excluded for missing
runtime dependencies.

| Policy | Passed | Pass Rate | Mean Latency |
| --- | ---: | ---: | ---: |
| trained-orchestrator | 1 / 3 | 0.3333 | 7507 ms |
| fixed-worker:ollama-cloud-qwen3-coder-480b | 0 / 3 | 0.0000 | 2897 ms |

Task-level details:

- `BigCodeBench-350`: router selected Qwen; both router and fixed baseline
  failed with the same response.
- `BigCodeBench-1039`: router selected Kimi; both Kimi and fixed Qwen failed,
  and Kimi was substantially slower.
- `BigCodeBench-339`: router selected GLM; GLM passed all tests while fixed
  Qwen failed.

## Decision

Keep the result as clean live evidence for GLM specialist routing on
`BigCodeBench-339`, but do not promote the policy. The positive is repeated and
useful, yet the sample is tiny and the router remains much slower on average.

The Qwen-small logits-head training path remains the right next model step, but
actual head training is still blocked locally by missing ML dependencies
(`torch`, `transformers`, `mlx`, and `mlx_lm`). Until those are installed or
GPU/MLX access is available, continue improving measured task-level data.
