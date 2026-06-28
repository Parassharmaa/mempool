# Adaptive Refresh Cycle

Run tag: `20260628-adaptive-refresh-cycle`

## What Changed

Added the first auditable adaptive-memory refresh-cycle builder:

- Module: `src/mempool/adaptive_refresh.py`
- CLI: `tools/build_adaptive_refresh_cycle.py`
- Test: `tests/test_adaptive_refresh.py`

The refresh cycle packages:

- immutable raw-trace pointer
- versioned distilled dataset
- candidate model and report
- policy gate result
- active-policy rollback point
- privacy/memory-scope manifest
- final promote/quarantine decision

## Artifact

Generated the first evaluated refresh cycle for the offline multi-head
candidate:

- Refresh cycle: `research/refreshes/20260628-m5-offline-multihead-refresh-cycle.json`
- Privacy manifest: `research/refreshes/20260628-m5-offline-multihead-privacy.json`
- Distilled dataset: `research/datasets/20260628-m5-small-orchestrator-substrate-50task.jsonl`
- Candidate model: `research/models/20260628-m5-offline-multihead-50task.json`
- Policy gate: `research/evals/results/20260628-m5-offline-multihead-50task-policy-gate.json`

## Result

All refresh guardrails passed:

- immutable raw trace pointer exists
- distilled dataset is versioned
- candidate was evaluated before promotion
- active registry provides a rollback point
- privacy manifest declares benchmark/general training scope
- no private raw text is included

The refresh decision is still `quarantine` because the policy gate quarantined
the candidate. Promotion is not allowed, and the active 23-task policy remains
unchanged.

## Interpretation

M6 now has an end-to-end evaluated refresh-cycle path. The system can package a
candidate memory/model update with privacy and rollback evidence, then refuse to
promote it when the gate fails. This is the correct safety behavior for frequent
or eventual hourly refreshes.

The current blocker is not refresh plumbing. It is model quality: the candidate
does not meet target-accuracy or latency-regret gates.

## Next Step

Use this refresh-cycle artifact as the standard wrapper for future memory
updates. The next useful experiment should improve the candidate model or
features, then regenerate the same cycle and compare decisions.
