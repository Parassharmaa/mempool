# Latency-Regret Refresh Gate

Run tag: `20260627-latency-regret-refresh-gate`

## Question

Can the adaptive refresh gate reject candidates that preserve some pass rate but
increase avoidable latency on validation?

## Change

`tools/policy_refresh_gate.py` now carries `mean_latency_regret_ms` from
leave-one-out reports and supports two optional thresholds:

- `--max-loo-latency-regret-ms`
- `--max-loo-latency-regret-increase-ms`

The default behavior remains compatible with earlier refreshes. Latency guards
only apply when the thresholds are supplied.

## Rechecked Candidate

The ten-task candidate refresh was regenerated with corrected nonnegative
latency-regret reports and re-gated with:

- max LOO latency regret: 1000 ms
- max LOO latency regret increase: 500 ms

Baseline active eight-task LOO:

- target accuracy: 0.50
- pass@1: 0.50
- mean latency regret: 522.125 ms

Ten-task candidate LOO:

- target accuracy: 0.40
- pass@1: 0.60
- mean latency regret: 1556.1 ms

Decision: quarantine.

Reasons:

- candidate target accuracy is below the minimum
- candidate latency regret increased by 1034.0 ms
- candidate latency regret exceeds the 1000 ms maximum

## Interpretation

The new guard protects against an important failure mode for a cheap live
orchestrator: a refresh can add more broadly solvable Qwen-latency examples yet
still damage validation routing by over-choosing slower regions or losing
specialist boundaries.

## Next Step

Train the next candidate against a latency-regret-aware objective or add
features that separate broad-pass latency-tie tasks from Kimi/GLM specialist
tasks, then use this gate before promotion.
