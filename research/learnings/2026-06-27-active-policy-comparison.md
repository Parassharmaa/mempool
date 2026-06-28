# 2026-06-27 Active Policy Comparison

## Question

Can the active learned policy be reported next to static router baselines in a
single benchmark artifact?

## Setup

- Dataset: `research/datasets/20260627-mixed-winner-8task-routing.jsonl`
- Active policy registry: `research/policies/active_policy.json`
- Report:
  `research/datasets/20260627-mixed-winner-8task-router-comparison-report.json`

## Result

`tools/train_router_baseline.py` now accepts `--active-policy-registry` and
adds an `active-logits-router` evaluation to the baseline report.

On the active eight-task dataset:

- active-logits-router: 8/8 target accuracy, 8/8 pass@1
- oracle-target: 8/8 target accuracy, 8/8 pass@1
- nearest-neighbor-router: 8/8 target accuracy, 8/8 pass@1
- nearest-neighbor-router-loo: 6/8 target accuracy, 7/8 pass@1
- strongest-worker: 2/8 target accuracy, 6/8 pass@1
- fastest-worker: 5/8 target accuracy, 5/8 pass@1
- family-router: 5/8 target accuracy, 5/8 pass@1

## Interpretation

The active learned policy now appears in the same comparison artifact as the
static baselines. This is a cleaner operational surface: a policy refresh changes
the registry, and the benchmark command evaluates the promoted learned policy.

The caveat is that active-logits-router and nearest-neighbor-router are both
training-set evaluations here. Leave-one-out remains the better sanity check for
generalization until we have a separate held-out task set.
