# Conditional Latency Calibration

Added latency calibration for logits-router artifacts:

- `tools/evaluate_latency_calibrated_logits_router.py`
- routing-record support in `src/mempool/latency_calibrated_router.py`
- tests:
  - `tests/test_latency_calibrated_router.py`
  - `tests/test_evaluate_latency_calibrated_logits_router.py`

Problem:

The replay-weighted and anchored refresh experiments improved pass rate, but
could not safely learn the all-pass latency rows as normal hard targets. A
global latency penalty was also too blunt: it reduced latency regret, but it
selected fast failing workers on base rows.

Experiment:

- base model:
  `research/models/20260627-mixed-winner-23task-heldout-hard-reward-t0p05-logits-router.json`
- dataset:
  `research/datasets/20260628-normal-offset16-contrast-29task-weight0p25-routing.jsonl`
- latency-only task ids:
  `research/evals/20260628-normal-offset16-contrast-task-ids.txt`
- report:
  `research/evals/20260628-conditional-latency-calibration-29task.json`
- model artifact:
  `research/models/20260628-conditional-latency-calibrated-logits-router-29task.json`

Selected conditional policy:

- apply latency calibration only to the six known all-pass latency rows
- keep the base logits-router top choice everywhere else
- source latency cost: 0.5 per second

Metrics on the 29-task candidate dataset:

| metric | base logits router | conditional latency calibration |
| --- | ---: | ---: |
| target accuracy | 0.6897 | 0.8966 |
| pass@1 | 0.8276 | 0.8276 |
| solvable pass@1 | 0.9231 | 0.9231 |
| latency regret | 562.3 ms | 178.8 ms |

Slice behavior:

- Base rows should not receive global latency calibration; doing so drops
  pass@1 from 0.7826 to 0.6522 on the base slice.
- All-pass latency rows benefit strongly; max latency calibration moves target
  accuracy from 0.0 to 1.0 and latency regret from 2736.0 ms to 0.0 on that
  slice.

Learning:

This is the first refresh-side mechanism in this sequence that improves target
accuracy and latency regret while preserving pass@1 on the mixed 29-task
candidate. The caveat is important: the allowlist currently comes from measured
outcomes, so it is an oracle diagnostic rather than a deployable runtime
classifier. The next step is to train or verify the condition "this task is safe
for latency optimization" from task features, cheap probes, or a verifier head.

Keep the active 23-task policy unchanged until the all-pass/latency-safe
condition can be predicted or verified without oracle labels.
