# Operational Refresh Reference

Run tag: `20260628-operational-refresh-reference`

## Question

Can refresh selection gate new router candidates against the best measured
operational policy, rather than only the previous raw logits-router LOO report?

## Change

- Added named operational-policy metric extraction to
  `tools/policy_refresh_gate.py`.
- Added `operational_reference` support to refresh decisions and
  `preserve_accuracy` threshold derivation.
- Threaded the same reference through `tools/select_logits_router_temperature.py`
  with:
  - `--operational-baseline-report`
  - `--operational-baseline-policy`
- Reran the 30-task live-augmented temperature sweep using
  `probe-gated-latency-calibrated-logits-router` as the operational reference.

## Result

The operational reference is now explicit in every refresh decision. For the
30-task live-augmented sweep, `preserve_accuracy` derived these thresholds from
the probe-gated policy:

- minimum LOO target accuracy: 0.8333
- minimum LOO solvable pass@1: 0.8519
- maximum LOO latency regret: 251.4 ms
- minimum LOO pass@1 vs strongest worker: strongest-worker pass@1 plus 0.0

All raw retrained candidates remained quarantined:

- temperature 0.05: target accuracy 0.7000, pass@1 0.7333, regret 687.7 ms
- temperature 0.10: target accuracy 0.7333, pass@1 0.8000, regret 1024.8 ms
- temperature 0.20: target accuracy 0.7000, pass@1 0.8000, regret 1544.4 ms
- temperature 0.50: target accuracy 0.6333, pass@1 0.7667, regret 1619.7 ms

The best raw retrain still improves pass@1 over the strongest single worker, but
it fails the operational bar because it gives up too much target accuracy and
latency regret compared with the probe-gated policy.

## Decision

Keep the operational-reference gate. Do not promote any raw live-augmented
logits refresh.

## Next

The next refresh candidate should not be another plain reward-temperature sweep.
It should train or select a conditional/probe-gated decision structure directly,
then use this operational-reference gate before promotion.
