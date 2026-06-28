# Multi-Head Latency-Aware Objective

Run tag: `20260628-multihead-latency-aware`

Goal: reduce the quarantined 50-task multi-head orchestrator's latency regret
without losing the solvable-task pass signal.

Change tested:

- Added an optional expected latency-regret penalty to the multi-head worker
  head.
- Trained weights `0.05`, `0.1`, `0.2`, `0.5`, and `1.0`.
- Gated the best latency candidate against the active 23-task policy.

Result:

- Unweighted 50-task multi-head LOO: target accuracy `0.62`, pass@1 `0.82`,
  solvable pass@1 `0.872`, latency regret `3609.63 ms`.
- Best latency-aware candidate, weight `0.5`: target accuracy `0.60`, pass@1
  `0.80`, solvable pass@1 `0.851`, latency regret `1693.48 ms`.
- Weight `1.0` did not improve regret further and lowered accuracy/pass@1.

Decision: quarantine latency-aware refresh.

Interpretation:

Latency-aware loss is useful: it cuts leave-one-out latency regret by more than
half while keeping solvable pass@1 above the current gate. It is not sufficient
for promotion because target accuracy still misses the gate and absolute latency
regret remains too high. The next step should combine this objective with better
specialist features or a selective verifier/fallback head, rather than relying
on latency pressure alone.

Artifacts:

- `research/refreshes/20260628-m5-latency-aware-selection.json`
- `research/evals/results/20260628-m5-latency-w0p5-policy-gate.json`
