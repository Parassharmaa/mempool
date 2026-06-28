# Current Status

`mempool` has reached the first trained task-level orchestrator checkpoint.

## Data

- Raw measured outcome rows on disk: 2029 before the latest active-rescue run.
- Unique task ids in measured outcomes: 199 before the latest active-rescue run.
- Clean current task-level substrate: 66 records.
- Current substrate target mix:
  - DeepSeek: 9
  - GLM: 4
  - Kimi: 12
  - Qwen: 41

## Current Task-Level Orchestrator

Artifact:

```text
research/models/20260628-m5-current-task-66task-multihead.json
```

Source substrate:

```text
research/datasets/20260628-m5-current-task-66task-substrate.jsonl
```

Training-set metrics:

- target accuracy: 0.6818
- pass@1: 0.8333
- solvable pass@1: 0.9649
- mean latency regret: 1575.7 ms

Leave-one-out metrics:

- target accuracy: 0.5606
- pass@1: 0.7576
- solvable pass@1: 0.8772
- mean latency regret: 2463.5 ms

Decision: trained and kept as a checkpoint, but not promoted as the active
policy.

## Next

Before turn-level training, improve task-level orchestrator reliability:

- collect more non-Qwen specialist targets
- reduce latency regret against the probe-gated operational reference
- improve leave-one-out target accuracy
- keep all refreshes gated and reversible

Turn-level agentic training should come after this task-level checkpoint is
stronger. The intended design is to predict each agentic turn's worker,
workflow, verifier, stop/repair/switch action, and memory-update decision from
trajectory state.

The turn-level substrate builder is now code-ready as a deferred path:
`tools/build_agentic_turn_substrate.py` converts sanitized trajectory summaries
into per-turn examples, while rejecting raw terminal output. It should remain a
data-contract scaffold until real multi-turn trajectories exist.

The current checkpoint can also be queried locally without retraining:

```bash
PYTHONPATH=src python3 tools/predict_multi_head_orchestrator.py \
  --model research/models/20260628-m5-current-task-66task-multihead.json \
  --prompt "Write Python code that reads files from a directory." \
  --task-family bigcodebench_hard \
  --categories filesystem,text \
  --libraries pathlib
```

The checkpoint can also route and prepare an OpenAI-compatible worker call:

```bash
PYTHONPATH=src python3 tools/run_orchestrated_prompt.py \
  --dry-run \
  --model research/models/20260628-m5-current-task-66task-multihead.json \
  --worker-pool research/evals/ollama_cloud_worker_pool_top4.json \
  --prompt "Write Python code that reads files from a directory." \
  --task-family bigcodebench_hard \
  --categories filesystem,text \
  --libraries pathlib
```
