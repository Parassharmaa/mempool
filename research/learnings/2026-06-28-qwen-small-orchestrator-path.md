# Qwen-Small Orchestrator Path

Run tag: `20260628-qwen-small-orchestrator-path`

## Clarification

The current linear logits router is not the final orchestrator. It is a baseline
and a data-loop check. The desired architecture is a small local model with
explicit routing heads.

The next neural milestone should use a Qwen-small style backbone when local
hardware allows it:

```text
task and compact state summary
  -> small language-model backbone
  -> decision hidden state
  -> worker logits
  -> workflow logits
  -> verifier probability
  -> abstain/fallback probability
  -> later: turn action and memory-update logits
```

## Training Order

1. Keep collecting task-level outcome rows and soft routing targets.
2. Train explicit heads on top of a frozen small backbone.
3. Compare against the current linear logits router on held-out task-level
   routing.
4. Only then try LoRA/adapters on the backbone.
5. Keep turn-level heads scaffolded but deferred until real multi-turn
   trajectories exist.

## Decision

Proceed toward a Qwen-small logits-head orchestrator, but do not skip the
task-level data-quality gate. If the small backbone does not beat the linear
router, the dataset is still too small or too noisy.

## Implementation Status

Added the source-level training path:

- `src/mempool/qwen_logits_orchestrator.py`
- `tools/train_qwen_logits_orchestrator.py`
- `tests/test_qwen_logits_orchestrator.py`

Generated training artifacts:

- `research/models/20260628-qwen-small-logits-orchestrator-plan.json`
- `research/datasets/20260628-qwen-small-logits-orchestrator-rows.jsonl`

The plan uses `Qwen/Qwen2.5-0.5B-Instruct` by default, freezes the backbone for
the first training pass, and trains explicit worker, workflow, verifier, and
abstain heads.

Attempted real training locally with `--train`, but the environment is missing
`torch` and `transformers`, so the guard stopped before model download or
training. We need to install the ML stack here or use GPU/MLX access for the
first real head-training run.

Follow-up readiness audit:

- `research/models/20260628-qwen-training-readiness.json`

The active environment is macOS arm64 with Python `3.14.4`; it has no
`torch`, `transformers`, `mlx`, or `mlx_lm`. For local training, create a
Python 3.11 or 3.12 environment and install `.[qwen-train]`. For a serious run,
prefer GPU or Apple MLX access.
