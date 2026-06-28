# Latency-Safe Head

Added a small classifier for the condition discovered by conditional latency
calibration: "is this routing row safe for latency-only optimization?"

Implementation:

- `src/mempool/latency_safe_head.py`
- `tools/train_latency_safe_head.py`
- `tests/test_latency_safe_head.py`

Label:

- positive if every worker in the measured row passes at `pass_rate >= 1.0`
- negative otherwise

Experiment:

- dataset:
  `research/datasets/20260628-normal-offset16-contrast-29task-weight0p25-routing.jsonl`
- router context:
  `research/models/20260627-mixed-winner-23task-heldout-hard-reward-t0p05-logits-router.json`
- default report:
  `research/evals/20260628-latency-safe-head-29task-report.json`
- sweep:
  `research/evals/20260628-latency-safe-head-29task-sweep.json`
- model:
  `research/models/20260628-latency-safe-head-29task.json`

Default run:

- in-sample accuracy: 1.0
- in-sample precision: 1.0
- in-sample recall: 1.0
- leave-one-out accuracy: 0.5517
- leave-one-out precision: 0.5
- leave-one-out recall: 0.5385

Threshold/positive-weight sweep:

- best leave-one-out precision: 0.625
- best leave-one-out accuracy: 0.6207
- true positives: 5
- false positives: 3
- false negatives: 8

Learning:

The latency-safe condition is not yet reliably learnable from the current
29-row dataset. The head can memorize the measured labels, but leave-one-out
performance is too weak for automatic runtime use. False positives are
especially dangerous because they would apply latency calibration to rows where
fast workers may fail.

Keep the conditional latency-calibration result as an oracle diagnostic, not as
an active deployable policy. The next acquisition should deliberately collect
matched controls:

1. all-pass latency rows across varied task categories,
2. near-neighbor rows where at least one worker fails,
3. repeated samples for both groups so the label is stable.

Only after the latency-safe head reaches high leave-one-out precision should it
replace the measured allowlist.
