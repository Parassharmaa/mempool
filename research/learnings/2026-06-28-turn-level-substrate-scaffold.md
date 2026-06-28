# Turn-Level Substrate Scaffold

Run tag: `20260628-turn-level-substrate-scaffold`

The project now has a schema-ready path for future agentic turn-level data.
This is intentionally not a training step yet.

## What Changed

- Added `mempool.agentic_turn_substrate`.
- Added `tools/build_agentic_turn_substrate.py`.
- Added tests for summary-only trajectories, repair/stop/memory labels, and raw
  terminal-output rejection.
- Added a tiny synthetic sanitized trajectory fixture under `research/evals/`.

## Decision

Keep this as preparation for the Terminal-Bench and trace-distillation phase.
The current active learning path remains task-level routing until the
task-level orchestrator is stronger and we have real, sanitized multi-turn
trajectories.
