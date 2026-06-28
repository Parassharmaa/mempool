# M5 Small-Orchestrator Substrate

Run tag: `20260628-m5-orchestrator-substrate`

## What Changed

Added a supervised substrate builder for the small trainable orchestrator path:

- Module: `src/mempool/small_orchestrator_substrate.py`
- CLI: `tools/build_small_orchestrator_substrate.py`
- Test: `tests/test_small_orchestrator_substrate.py`

The builder converts repeated routing records into one JSONL example per task
with both structured labels and chat-style messages for future offline
fine-tuning.

Each example includes targets for the M5 multi-head action surface:

- `worker_distribution`
- `workflow_kind`
- `verifier_probability`
- `abstain_probability`

## Artifact

Generated the first 50-row small-orchestrator substrate from the threshold
dataset:

- Input routing dataset:
  `research/datasets/20260627-expanded-profile-wave4-50task-routing.jsonl`
- Multi-head contract:
  `research/models/20260627-active-multi-head-orchestrator-contract.json`
- Output substrate:
  `research/datasets/20260628-m5-small-orchestrator-substrate-50task.jsonl`
- Manifest:
  `research/datasets/20260628-m5-small-orchestrator-substrate-50task-manifest.json`

Manifest summary:

- Records: 50
- Workflow labels: 47 direct, 3 verify-then-fallback
- Abstain-positive rows: 3
- Target mix: Qwen 30, DeepSeek 9, Kimi 8, GLM 3
- Mean verifier probability: 0.390

## Interpretation

The project now has a concrete offline fine-tuning substrate for a compact
orchestrator. This does not promote the quarantined 50-task router and does not
claim small-orchestrator quality yet. It creates the missing bridge between the
audited routing ledger and a future trainable backbone.

The current labels are conservative. Direct workflow labels are used for
solvable rows. All-fail rows become verify-then-fallback/abstain examples.
Verifier probability is derived from worker-distribution uncertainty until
trajectory-level verifier labels exist.

## Next Step

Train a tiny offline candidate against this substrate, or export it into the
chosen local fine-tuning framework, then evaluate it with the existing policy
gate before any active-policy promotion.
