# Logits-Head Orchestrator And Cloud Workers

## Change

Updated the architecture plan toward a low-latency orchestrator that emits
action logits directly from a compact backbone representation.

Added a real Ollama Cloud worker-pool config:

- `research/evals/ollama_cloud_worker_pool.json`

The key is stored locally in ignored `.env` as `OLLAMA_API_KEY`; it is not
committed.

## Cloud Worker Pool

The live Ollama Cloud catalog exposed several top worker candidates. The first
cloud pool uses:

- `glm-5.2`
- `deepseek-v4-pro`
- `deepseek-v3.2`
- `kimi-k2.7-code`
- `qwen3-coder:480b`

These are workers in the pool, not the orchestrator itself. The orchestrator
should learn when to call them.

## Architecture Learning

The fast orchestrator path should not be a text-generating planner. It should:

1. encode the task with a compact backbone
2. read a hidden state at a fixed decision position
3. apply lightweight heads to emit logits
4. dispatch to a worker or workflow action

Training should use measured worker rewards converted into soft routing targets,
then minimize KL divergence from those targets to the predicted distribution.

## Harness Position

The current harness is a custom BigCodeBench single-step evaluator. It is the
right first harness for worker-selection labels because it gives objective
pass/fail, latency, and failure-mode records.

OpenCode/Codex-style harnesses are still important, but they belong to the next
stage: end-to-end agentic trajectories for optimizing multi-turn routing,
verification, tool use, and workflow selection.

## Next Step

Run a tiny cloud smoke benchmark on the eligible BigCodeBench pool, starting
with one or two tasks and the top cloud workers, to collect positive routing
labels before training the logits-head orchestrator.
