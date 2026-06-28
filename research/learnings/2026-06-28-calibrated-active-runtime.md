# Calibrated Active Runtime

Run tag: `20260628-calibrated-active-runtime`

## Question

The latency-calibrated multi-head router cleared the refresh gate against the 50-task multi-head baseline, but it was not executable through the active-policy runtime. Can the active-policy path load and evaluate both existing logits-router policies and the new calibrated multi-head wrapper without breaking router-only acquisition tools?

## Change

Extended `tools/evaluate_active_policy.py` with a generic active-policy loader:

- `load_active_router(...)` remains router-only for existing acquisition tools.
- `load_active_policy(...)` loads the raw active model payload and registry entry.
- `evaluate_active_policy_payload(...)` dispatches by `model_type`.
- Supported model types now include:
  - `logits-router`
  - `linear-softmax-logits-router`
  - `latency-calibrated-multi-head-router`

The calibrated runtime path loads the wrapper artifact, resolves its base multi-head model, reads the substrate path embedded in the model artifact, predicts with the multi-head model, and applies the transparent latency calibration.

The CLI now prints concise metrics while still writing full per-task examples to the output JSON.

## Compatibility Checks

Current active logits-router policy still evaluates successfully:

- Registry: `research/policies/active_policy.json`
- Output: `research/evals/results/20260628-active-policy-runtime-compat-eval.json`
- Policy type: `linear-softmax-logits-router`
- Runtime target accuracy: `0.870`
- Runtime pass@1: `0.783`
- Runtime solvable pass@1: `0.900`
- Runtime latency regret: `501.1 ms`

A simulated active registry for the calibrated multi-head wrapper also evaluates successfully:

- Registry: `research/policies/20260628-latency-calibrated-router-simulated-active.json`
- Output: `research/evals/results/20260628-latency-calibrated-router-simulated-active-eval.json`
- Policy type: `latency-calibrated-multi-head-router`
- Runtime target accuracy: `0.778`
- Runtime pass@1: `0.889`
- Runtime solvable pass@1: `0.941`
- Runtime latency regret: `443.1 ms`

These runtime metrics are in-sample evaluations of the installed model artifacts. They are not replacements for the held-out/LOO refresh gates.

## Promotion Decision

The calibrated candidate remains runtime-ready but not active-promoted.

It still promotes against the 50-task multi-head baseline:

- Gate: `research/evals/results/20260628-latency-calibrated-router-offset76-w0p5-policy-gate.json`
- Decision: `promote`

But it quarantines against the actual active policy registry baseline:

- Normalized active baseline report: `research/evals/results/20260628-active-policy-registry-loo-report.json`
- Gate: `research/evals/results/20260628-latency-calibrated-router-vs-active-policy-gate.json`
- Decision: `quarantine`
- Reasons:
  - LOO target accuracy drop `0.153` exceeded the allowed `0.080`
  - LOO latency regret increase `795.5 ms` exceeded the allowed `250.0 ms`

So the active registry should remain unchanged for now.

## Interpretation

The calibrated multi-head wrapper is now executable through the same evaluation CLI as the existing active logits router. That removes the runtime blocker from the previous run.

The promotion blocker is now metric-based, not infrastructure-based: the candidate is useful for the larger 54-task multi-head track, but the current active 23-task policy still has stronger stored LOO target accuracy and lower latency regret under the current gate.

## Next Step

Keep the active policy unchanged. Use the calibrated runtime path for candidate evaluation and continue acquiring guarded BigCodeBench rows. The next active promotion should either:

- improve the calibrated 54-task candidate enough to pass against the active registry baseline, or
- use an explicit policy-family promotion rule that treats the larger calibrated multi-head track as a separate experimental lane instead of replacing the 23-task logits-router active policy.
