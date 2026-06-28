# Solvability-Screened Live Validation

Attempted a solvability-screened live validation batch for the frozen
probe-gated policy.

Frozen policy under validation:

- `research/policies/20260628-probe-gated-latency-calibrated-policy.json`

Candidate selection:

- broad candidate report:
  `research/evals/20260628-solvability-screened-live-candidates12-selection.json`
- evaluator-friendly filtered candidates:
  `research/evals/20260628-solvability-screened-live-stdlib-candidates12-tasks.json`
  and
  `research/evals/20260628-solvability-screened-live-stdlib-candidates12-selection.json`

Screen run:

- worker:
  `ollama-cloud-qwen3-coder-480b`
- outcomes:
  `research/evals/results/20260628-solvability-screened-live-qwen-screen1.jsonl`
- summary:
  `research/evals/results/20260628-solvability-screened-live-qwen-screen1-summary.json`
- graduated positives:
  `research/evals/20260628-solvability-screened-live-qwen-positive-tasks.json`

Result:

- 12 fresh candidate tasks
- 1 Qwen sample per task
- 0 passing samples
- 0 graduated tasks

Learning:

The screen prevented another repeated top-4 run on an all-fail batch, so the
process change is useful. However, the current fresh standard-library candidate
pool is still too hard or mismatched for the present prompt/evaluator setup.
Low environment risk and standard-library dependencies are not enough to
predict solvability.

Decision:

Discard this as policy-validation evidence. Keep the solvability-screen step as
a required gate before future live repeated top-4 validation.

Next step:

Use a known-positive or near-positive acquisition source instead of generic
fresh task novelty:

1. seed candidate selection from tasks already solved by Qwen/Kimi/GLM in prior
   screens,
2. require one positive screen sample before top-4 repeat,
3. only then evaluate the frozen probe-gated policy.
