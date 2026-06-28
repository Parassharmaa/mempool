# Gated Fallback Operational Gate

Run tag: `20260628-gated-fallback-operational-gate`

## Question

Can a conditional fallback policy become the next promotable candidate when
gated against the best measured probe-gated operational policy?

## Change

- Added `tools/gate_operational_policy.py`, a gate for evaluated workflow
  policies that do not have leave-one-out router metrics.
- Selected a margin-gated fallback policy on the 30-task live-augmented dataset
  while requiring existing regression slices to pass.
- Gated the selected fallback policy against
  `probe-gated-latency-calibrated-logits-router` from
  `research/evals/20260628-live-augmented-30task-baseline-report.json`.

## Result

The selected gated fallback policy uses margin `0.15` with `max_attempts=2` and
passes the regression-slice manifest.

On the 30-task live-augmented dataset:

- candidate gated fallback: pass@1 0.8333, solvable pass@1 0.9259, target
  accuracy 0.8000, mean latency regret 1815.3 ms
- probe-gated reference: pass@1 0.7667, solvable pass@1 0.8519, target accuracy
  0.8333, mean latency regret 251.4 ms

The candidate improves pass@1 by 0.0667 and solvable pass@1 by 0.0741, and it
beats strongest-worker pass@1 0.6667. It is still quarantined because latency
regret increases by 1563.9 ms against the probe-gated reference.

## Decision

Keep:

- operational workflow-policy gate
- selected gated fallback policy artifact
- regression-slice selection artifact

Do not promote:

- margin-gated fallback policy as the active operational policy

## Next

Train or select a value-aware fallback policy that predicts expected rescue
benefit relative to extra latency. A pure router-margin rule can find rescues,
but it takes too many expensive second attempts.
