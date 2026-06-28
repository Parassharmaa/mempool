# Latency-Safe Reliability Features

Tested whether historical per-worker reliability features improve the
latency-safe head.

Implementation:

- `src/mempool/latency_safe_head.py`
  - `worker_reliability_context`
  - reliability pass/stable-rate feature block
  - leave-one-out computes reliability only from training rows
- `tools/train_latency_safe_head.py`
  - `--use-reliability-features`
- `tests/test_latency_safe_head.py`

Experiment:

- dataset:
  `research/datasets/20260628-latency-safe-head-37task-routing.jsonl`
- router model:
  `research/models/20260627-mixed-winner-23task-heldout-hard-reward-t0p05-logits-router.json`
- report:
  `research/evals/20260628-latency-safe-head-reliability-feature-report.json`

Result:

| feature set | best LOO precision | best LOO accuracy | best LOO recall |
| --- | ---: | ---: | ---: |
| prompt + router confidence | 0.6000 | 0.6216 | 0.3750 |
| + worker reliability features | 0.6000 | 0.6216 | 0.3750 |

Learning:

Historical worker reliability features, as currently defined, do not improve
the latency-safe condition. The head still cannot reliably distinguish all-pass
latency-safe rows from nearby unsafe controls under leave-one-out validation.

Decision:

Discard this as a promotion path. The code path remains useful for future
feature experiments, but the metric says this feature family is not sufficient.

Next step:

Move from static task/router features to an evidence-producing condition:

1. cheap probe outcomes before applying latency calibration, or
2. a verifier-style worker-agreement head that observes partial agreement
   signals rather than predicting all-pass safety from prompt features alone.
