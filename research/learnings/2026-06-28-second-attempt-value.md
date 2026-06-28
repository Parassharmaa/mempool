# Second-Attempt Value Diagnostic

Run tag: `20260628-second-attempt-value-head`

Goal: define a better training target for the fallback/verifier head than
generic uncertainty. The new target estimates whether trying the next ranked
worker is worth the added latency:

`value = solve_gain - latency_cost`

Change tested:

- Added reusable second-attempt value labels.
- Added an oracle value-gated fallback diagnostic over the latency-aware
  multi-head leave-one-out predictions.
- Gated the diagnostic with the same refresh criteria used for policy
  candidates.

Result:

- Base latency-aware `w0.5`: target accuracy `0.60`, pass@1 `0.80`, solvable
  pass@1 `0.851`, latency regret `1693.48 ms`.
- Blunt high-pass fallback from the prior run: pass@1 `0.86`, solvable pass@1
  `0.915`, latency regret `4560.08 ms`.
- Oracle second-attempt value diagnostic: target accuracy `0.62`, pass@1
  `0.86`, solvable pass@1 `0.915`, latency regret `2925.26 ms`.

Decision: quarantine diagnostic.

Interpretation:

Second-attempt value is a better fallback target than generic verifier
probability: it preserves the coverage gain while reducing the fallback regret
substantially. It is still not enough for promotion because target accuracy
remains too low and latency regret remains far above the active-policy gate.

Next step:

Train a real fallback/value head to predict this label from model-visible
features, then combine it with improved specialist routing features. The value
head alone cannot fix the underlying DeepSeek/GLM target misses.

Artifacts:

- `src/mempool/second_attempt_value.py`
- `tools/evaluate_second_attempt_value.py`
- `research/refreshes/20260628-m5-second-attempt-value-diagnostic.json`
- `research/evals/results/20260628-m5-second-attempt-value-diagnostic-policy-gate.json`
