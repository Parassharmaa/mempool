# Profiled Adaptive Refresh Cycle

## Question

Does the adaptive refresh cycle preserve and expose the promotion profile used
by the policy gate?

## Result

Yes. Adaptive refresh cycle artifacts now carry `promotion_profile` at the
top level and inside the promotion block.

Generated a profiled cycle for the router-miss repeat candidate:

- `research/refreshes/20260628-router-miss-repeat-24task-profiled-policy-gate.json`
- `research/refreshes/20260628-router-miss-repeat-24task-profiled-privacy.json`
- `research/refreshes/20260628-router-miss-repeat-24task-profiled-refresh-cycle.json`

The cycle uses `promotion_profile: preserve_accuracy`.

All adaptive-memory guardrails passed:

- immutable raw traces
- versioned distilled dataset
- evaluation before promotion
- rollback point
- privacy filter
- separate user memory
- candidate artifact

The final cycle decision remains `quarantine` because the profiled gate rejects
the candidate's target-accuracy drop.

## Decision

Keep the profile-aware adaptive refresh cycle. Future refresh artifacts can now
show whether they used tolerant exploration or accuracy-preserving promotion
rules without requiring a reader to infer that from numeric thresholds.
