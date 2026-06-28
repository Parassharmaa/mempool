# Orchestrated Prompt-Set Comparison

Run tag: `20260628-orchestrated-promptset-comparison`

The live route-then-execute path now supports a small prompt-set comparison
between the trained orchestrator and a fixed-worker baseline.

## Result

Prompt set: `research/evals/20260628-orchestrated-promptset-comparison-prompts.json`

Artifacts:

- Full comparison: `research/evals/20260628-orchestrated-promptset-comparison.json`
- Outcome rows: `research/evals/20260628-orchestrated-promptset-comparison-outcomes.jsonl`

Summary:

- Prompts: 3
- Executions: 6
- Fixed baseline: `ollama-cloud-qwen3-coder-480b`
- Orchestrator-selected workers:
  - `ollama-cloud-qwen3-coder-480b`: 3 / 3
- Fixed-worker responses present: 3 / 3
- Orchestrator responses present: 3 / 3
- Mean latency:
  - trained orchestrator policy: 2543.0 ms
  - fixed Qwen baseline: 1911.0 ms

## Decision

Keep the prompt-set comparison harness. This run validates the dataset path and
comparison shape, but it does not demonstrate routing diversity because the
orchestrator selected the fixed baseline worker for every prompt. The next
prompt set should deliberately include tasks near known non-Qwen decision
regions, or use benchmark-backed prompts where GLM, Kimi, or DeepSeek have
measured wins.
