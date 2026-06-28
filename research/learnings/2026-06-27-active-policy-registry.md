# 2026-06-27 Active Policy Registry

## Question

Can a promoted policy refresh become an auditable active model with rollback
bookkeeping?

## Setup

- Refresh manifest:
  `research/refreshes/20260627-mixed-winner-8task-refresh.json`
- Active policy registry:
  `research/policies/active_policy.json`
- Tool: `tools/policy_registry.py`

## Result

`tools/policy_registry.py` now supports:

- `apply-refresh`: accepts only `decision=promote` refresh manifests, sets the
  candidate as active, stores the prior active policy as previous, and appends a
  history entry.
- `rollback`: swaps active and previous policies and appends a rollback history
  entry.

The current active policy is:

- Model: `research/models/20260627-mixed-winner-8task-logits-router.json`
- Dataset: `research/datasets/20260627-mixed-winner-8task-routing.jsonl`
- LOO target accuracy: 0.75
- LOO pass@1: 0.75
- Target workers: Qwen, Kimi, GLM

The promotion history preserves the warning from the refresh gate:

```text
candidate LOO target accuracy dropped by 0.083
```

## Decision

Adaptive refresh now has a minimal lifecycle:

1. Train a logits router.
2. Gate it against a baseline.
3. Promote or quarantine it with reasons.
4. Record the promoted policy as active with rollback state.

The next step is to make runtime routing load the active policy registry by
default, so evaluations can compare the active learned policy against static
baselines without hardcoding model paths.
