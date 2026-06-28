# Multi-Head Selective Fallback

Run tag: `20260628-multihead-selective-fallback`

Goal: test whether the latency-aware 50-task multi-head orchestrator can recover
missed solvable tasks by spending a second worker attempt selectively.

Change tested:

- Added a multi-head gated fallback evaluator that uses:
  - worker-distribution first/second margin,
  - verifier probability,
  - at most two worker attempts.
- Added a threshold selector for existing multi-head leave-one-out reports.
- Regenerated the latency-aware `w0.5` report so LOO predictions include full
  worker distributions.

Result:

- Base latency-aware `w0.5` LOO: target accuracy `0.60`, pass@1 `0.80`,
  solvable pass@1 `0.851`, latency regret `1693.48 ms`.
- High-pass fallback point: target accuracy `0.58`, pass@1 `0.86`, solvable
  pass@1 `0.915`, latency regret `4560.08 ms`.
- Lowest-regret pass-gain point: target accuracy `0.62`, pass@1 `0.82`,
  solvable pass@1 `0.872`, latency regret `1765.87 ms`.

Decision: quarantine fallback refresh.

Interpretation:

Selective fallback is useful for coverage but currently too blunt for
promotion. The high-pass policy spends too much latency. The low-regret policy
is a cleaner Pareto point, but it still misses target-accuracy and absolute
latency-regret promotion thresholds.

Next step:

Train the verifier/fallback head against explicit second-attempt value, not only
generic verifier probability. The label should estimate expected solve gain
minus latency regret for the next ranked worker.

Artifacts:

- `research/refreshes/20260628-m5-latency-w0p5-fallback-selection.json`
- `research/policies/20260628-m5-latency-w0p5-fallback-policy.json`
- `research/evals/results/20260628-m5-latency-w0p5-fallback-highpass-policy-gate.json`
- `research/evals/results/20260628-m5-latency-w0p5-fallback-lowregret-policy-gate.json`
