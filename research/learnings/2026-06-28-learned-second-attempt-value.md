# Learned Second-Attempt Value Head

Run tag: `20260628-learned-second-attempt-value`

Goal: train a real fallback/value head from the oracle second-attempt value
labels instead of relying on generic verifier probability.

Change tested:

- Added a weighted logistic second-attempt value head.
- Features are model-visible: top/second probabilities, margin,
  verifier/abstain probabilities, and top/second worker ids.
- Trained on the latency-aware 50-task multi-head leave-one-out predictions.
- Selected threshold by pass/accuracy/regret tradeoff and gated the candidate.

Result:

- Base latency-aware `w0.5`: target accuracy `0.60`, pass@1 `0.80`, solvable
  pass@1 `0.851`, latency regret `1693.48 ms`.
- Oracle value-gated fallback: target accuracy `0.62`, pass@1 `0.86`,
  solvable pass@1 `0.915`, latency regret `2925.26 ms`.
- Learned value head: target accuracy `0.60`, pass@1 `0.86`, solvable pass@1
  `0.915`, latency regret `3311.74 ms`.

Decision: quarantine learned value head.

Interpretation:

The value target is learnable enough to recover the solve-coverage gain, and it
beats the blunt high-pass fallback latency. It does not match the oracle and
does not fix the upstream target routing errors. The active blocker for
promotion remains specialist worker selection, especially DeepSeek/GLM target
misses, not merely fallback gating.

Next step:

Add specialist-routing features or acquire more DeepSeek/GLM-positive rows,
then rerun latency-aware training plus the learned value head. The value head
should stay as an auxiliary action head, not replace the worker router.

Artifacts:

- `research/models/20260628-m5-second-attempt-value-head.json`
- `research/evals/results/20260628-m5-second-attempt-value-head-report.json`
- `research/evals/results/20260628-m5-second-attempt-value-head-policy-gate.json`
