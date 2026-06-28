# Local Orchestrator Inference

Run tag: `20260628-local-orchestrator-inference`

The current task-level checkpoint can now be loaded as a local routing policy
without retraining.

## What Changed

- Added `mempool.orchestrator_runtime`.
- Added `tools/predict_multi_head_orchestrator.py`.
- Added tests for model loading, prompt feature extraction, and prediction
  output.
- Saved an example prediction at
  `research/evals/20260628-local-orchestrator-inference-example.json`.

## Decision

Keep this runtime path. It does not call workers yet; it only emits the routing
decision that a worker executor can consume later. That makes the trained
checkpoint usable while keeping evaluation and execution concerns separate.
