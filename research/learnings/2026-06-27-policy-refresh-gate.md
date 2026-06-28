# 2026-06-27 Policy Refresh Gate

## Question

Can router updates be gated before promotion, so adaptive refreshes do not
blindly accept every newly trained logits-head artifact?

## Setup

- Baseline report:
  `research/datasets/20260627-mixed-winner-6task-logits-router-report.json`
- Baseline dataset:
  `research/datasets/20260627-mixed-winner-6task-routing.jsonl`
- Candidate report:
  `research/datasets/20260627-mixed-winner-8task-logits-router-report.json`
- Candidate dataset:
  `research/datasets/20260627-mixed-winner-8task-routing.jsonl`
- Refresh manifest:
  `research/refreshes/20260627-mixed-winner-8task-refresh.json`

## Result

`tools/policy_refresh_gate.py` now writes a promote/quarantine decision based on
leave-one-out accuracy, dataset size, and target-worker diversity.

The eight-task candidate was promoted with a warning:

- Baseline LOO target accuracy: 0.8333333333333334
- Candidate LOO target accuracy: 0.75
- Allowed drop: 0.1
- Candidate drop: 0.083
- Baseline target workers: 2
- Candidate target workers: 3

The decision is `promote`, with warning:

```text
candidate LOO target accuracy dropped by 0.083
```

## Decision

This is the first adaptive refresh mechanism. It allows richer datasets to be
promoted when they add target diversity and remain within a bounded LOO
regression, while preserving reasons and warnings for auditability.

Next, policy refresh should gain rollback bookkeeping: active model pointer,
previous model pointer, and an explicit rollback command when a later refresh is
quarantined or underperforms.
