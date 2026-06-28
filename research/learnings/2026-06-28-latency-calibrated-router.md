# Latency-Calibrated Router

Run tag: `20260628-latency-calibrated-router`

## Question

The offset76 54-task multi-head candidate improved target accuracy and pass@1, but failed the policy gate because held-out latency regret rose to about `2592 ms`. Can a transparent calibration layer reduce latency regret without retraining the worker head or spending another worker batch?

## Change

Added a latency-calibrated worker choice layer:

- Input: multi-head worker probability distribution plus measured worker latency profile for the task record.
- Utility: `worker_probability - latency_cost_per_second * latency_seconds`.
- Guardrails: optional minimum probability and minimum ratio to the top predicted probability.
- Output: selected worker, utility-ranked eligible workers, latency/pass metrics, and full examples.

This is not a fallback-after-failure policy. It changes the first worker choice when another likely worker is much cheaper. That addresses the failure mode found in the offset76 run: the router was learning solvability faster than latency discipline.

## Artifacts

- Implementation: `src/mempool/latency_calibrated_router.py`
- CLI: `tools/evaluate_latency_calibrated_router.py`
- Tests: `tests/test_latency_calibrated_router.py`
- Calibrated policy artifact: `research/models/20260628-latency-calibrated-router-offset76-w0p5.json`
- Evaluation report: `research/evals/results/20260628-latency-calibrated-router-offset76-w0p5.json`
- Policy gate: `research/evals/results/20260628-latency-calibrated-router-offset76-w0p5-policy-gate.json`

## Result

The best calibration used:

- `latency_cost_per_second`: `0.01`
- `min_probability_ratio`: `0.0`
- `min_probability`: `0.0`

Against the raw offset76 `w0.5` multi-head LOO predictions:

- Raw LOO target accuracy: `0.611`
- Calibrated LOO target accuracy: `0.630`
- Raw LOO pass@1: `0.815`
- Calibrated LOO pass@1: `0.815`
- Raw LOO solvable pass@1: `0.863`
- Calibrated LOO solvable pass@1: `0.863`
- Raw LOO mean latency: `6981.9 ms`
- Calibrated LOO mean latency: `5686.5 ms`
- Raw LOO mean latency regret: `2592.0 ms`
- Calibrated LOO mean latency regret: `1296.6 ms`
- Choices changed from the top raw prediction: `2/54`

The existing refresh gate promoted the calibrated candidate against the 50-task baseline:

- Baseline LOO target accuracy: `0.600`
- Candidate LOO target accuracy: `0.630`
- Baseline LOO pass@1: `0.800`
- Candidate LOO pass@1: `0.815`
- Baseline LOO solvable pass@1: `0.851`
- Candidate LOO solvable pass@1: `0.863`
- Baseline LOO latency regret: `1693.5 ms`
- Candidate LOO latency regret: `1296.6 ms`

Gate decision: `promote`.

## Runtime Caveat

Do not apply this gate directly to `research/policies/active_policy.json` yet. The active-policy runtime currently loads `LogitsRouter` artifacts, while this candidate is a calibrated multi-head wrapper. Applying it without a runtime loader update would make active-policy evaluation incompatible.

For now, treat this as a metric-promoted candidate artifact, not the installed active policy.

## Interpretation

The result confirms that the current multi-head worker probabilities contain useful latency-correctable signal. A tiny transparent calibration layer can fix the specific offset76 latency-regret failure without hurting pass rate.

This also suggests the next architecture step: make latency calibration a first-class runtime policy wrapper around the multi-head orchestrator, then teach the active-policy registry and evaluator to load policy types instead of assuming every active model is a logits router.

## Next Step

Add a runtime loader/evaluator for `latency-calibrated-multi-head-router` artifacts. Once it can evaluate through the same active-policy path as existing logits routers, re-run the promotion gate and then decide whether to update the active registry.
