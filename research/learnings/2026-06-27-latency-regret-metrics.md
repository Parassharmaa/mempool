# Latency Regret Metrics

Run tag: `20260627-latency-regret-metrics`

## Question

Can router evaluation make latency misses visible without hiding pass-rate or
specialist-boundary failures?

## Change

Router evaluations now report:

- `mean_target_latency_ms`
- `mean_latency_regret_ms`

Latency regret is nonnegative extra latency relative to the empirical target
worker. If a policy chooses a faster non-target worker, the latency regret is
zero and the failure is still captured by target accuracy and pass@1.

The metric is emitted for:

- family router
- nearest-neighbor router
- strongest-worker and fastest-worker baselines
- oracle target
- active logits router
- logits-router training and leave-one-out reports

## Findings

The held-out two-task diagnostic now shows the active policy's latency miss
directly:

- target worker: Qwen on both rows
- active predictions: Qwen, Kimi
- pass@1: 2/2
- target accuracy: 1/2
- mean target latency: 2372.0 ms
- mean active latency: 4630.5 ms
- mean latency regret: 2258.5 ms

On the ten-task merged diagnostic set, the active policy still solves 9/10 and
matches 7/10 targets, but carries 1138.4 ms mean latency regret. The quarantined
ten-task candidate remains rejected because its leave-one-out target accuracy is
0.40.

Regenerating the active eight-task report changed the current active
leave-one-out metadata to 0.50 target accuracy and 0.50 pass@1. The active
registry metadata was updated to match the current report, while the promotion
history remains unchanged.

## Interpretation

Pass@1, target accuracy, and latency regret need to be read together:

- pass@1 protects capability
- target accuracy protects the empirical reward target
- latency regret measures avoidable slowness when the chosen worker is slower
  than the target

This gives the next refresh objective a sharper target: reduce latency regret on
broad-pass tasks without losing sparse Kimi/GLM specialist regions.

## Next Step

Add a latency-aware refresh criterion or train a router variant that uses
latency-regret-sensitive validation, then compare it against the current active
policy before any promotion.
