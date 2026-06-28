# Value-Aware Fallback Gate

Run tag: `20260628-value-aware-fallback-gate`

## Question

Can a learned second-attempt value head improve on margin-gated fallback by
taking rescue value and extra latency into account?

## Change

- Fixed learned second-attempt value evaluation so it uses the learned value
  head for fallback decisions.
- Removed leakage from value-head features: the model no longer sees whether
  the second worker actually passed.
- Added `tools/select_second_attempt_value_policy.py` to train value-aware
  fallback candidates, sweep latency costs and thresholds, require regression
  slices to pass, and write a policy artifact.

## Result

The selector found a regression-passing learned value candidate:

- latency cost: 0.05
- threshold: -0.25
- pass@1: 0.8333
- solvable pass@1: 0.9259
- target accuracy: 0.7667
- mean latency regret: 2087.4 ms
- fallback rate: 0.7143

Against the probe-gated operational reference:

- reference pass@1: 0.7667
- reference solvable pass@1: 0.8519
- reference target accuracy: 0.8333
- reference mean latency regret: 251.4 ms

The learned value head is quarantined. It improves pass@1 and solvable pass@1,
but target accuracy drops by 0.0667 and latency regret increases by 1836.0 ms.

## Decision

Keep:

- non-leaky learned value-head evaluation
- value-aware fallback selector
- value-aware fallback policy artifact

Do not promote:

- current learned value-aware fallback candidate

## Next

The current training data is still too sparse for a learned value head to beat
the probe-gated operational reference. Mine or synthesize more fallback
opportunities, especially cases where the second attempt is useful but cheap,
then retrain with an operational-gate-aware selection objective.
