# Multi-Head Target Balance Experiment

Run tag: `20260628-multihead-feature-objective`

Goal: test whether the quarantined 50-task multi-head orchestrator failed
because the worker target mix is imbalanced toward Qwen.

Change tested:

- Added optional inverse-frequency target balancing to
  `train_multi_head_orchestrator`.
- Trained balance powers `0.25`, `0.5`, `0.75`, and `1.0` on the 50-task
  substrate.
- Gated the least-bad balanced candidate against the active 23-task policy.

Result:

- Unbalanced 50-task multi-head LOO: target accuracy `0.62`, pass@1 `0.82`,
  solvable pass@1 `0.872`, latency regret `3609.63 ms`.
- Best balanced candidate, power `0.25`: target accuracy `0.54`, pass@1 `0.80`,
  solvable pass@1 `0.851`, latency regret `4364.39 ms`.
- Stronger balancing made the result worse.

Decision: discard balanced refresh.

Interpretation:

Naive rare-target balancing overcorrects. It does not solve the DeepSeek/GLM
specialist misses and it hurts broad-pass latency choices. The next improvement
should not be class balancing alone. Better options are:

- richer specialist features for DeepSeek/GLM cases,
- explicit latency-regret-aware loss,
- held-out hard-slice weighting that focuses only on solvable misses,
- or a verifier/fallback head that spends a second attempt selectively.

Artifacts:

- `research/refreshes/20260628-m5-target-balance-selection.json`
- `research/evals/results/20260628-m5-balanced-p0.25-policy-gate.json`
