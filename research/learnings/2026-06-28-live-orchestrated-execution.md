# Live Orchestrated Execution

Run tag: `20260628-live-orchestrated-execution`

The trained task-level orchestrator has now selected a live Ollama-compatible
cloud worker and executed a prompt through that selected worker.

## Result

- Prompt: `Return one concise sentence saying hello from mempool.`
- Selected worker: `ollama-cloud-qwen3-coder-480b`
- Selected model: `qwen3-coder:480b`
- Worker distribution:
  - `ollama-cloud-qwen3-coder-480b`: 0.2912
  - `ollama-cloud-kimi-k2.7-code`: 0.2456
  - `ollama-cloud-deepseek-v4-pro`: 0.2324
  - `ollama-cloud-glm-5.2`: 0.2307
- Workflow: `direct`
- Verifier probability: 0.4986
- Abstain probability: 0.3872
- Latency: 2057 ms
- Response: `Hello from mempool!`

Artifacts:

- Full execution: `research/evals/20260628-live-orchestrated-execution.json`
- Flattened outcome row: `research/evals/20260628-live-orchestrated-execution-outcome.jsonl`

## Decision

Keep this as the first live route-then-execute proof. It is not a benchmark
score, but it proves the trained checkpoint can select a worker, resolve that
worker through the provider-neutral pool config, execute through the
OpenAI-compatible adapter, and emit a reusable outcome-style row.
