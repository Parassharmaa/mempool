# Probe-Gated Latency Calibration

Integrated latency-safe probe agreement into the logits-router latency
calibration evaluator.

Implementation:

- `tools/evaluate_latency_calibrated_logits_router.py`
  - added `--probe-worker-ids`
  - added `--probe-mode`
  - added `--probe-min-pass-rate`
  - derives the conditional calibration task set from probe pass agreement
- `src/mempool/latency_calibrated_router.py`
  - preserves the original router top choice through one-hot conditional
    predictions so `changed_from_top` remains meaningful
- `tests/test_evaluate_latency_calibrated_logits_router.py`

Experiment:

- dataset:
  `research/datasets/20260628-latency-safe-head-37task-routing.jsonl`
- base router:
  `research/models/20260627-mixed-winner-23task-heldout-hard-reward-t0p05-logits-router.json`
- reports:
  - `research/evals/20260628-probe-gated-latency-calibration-deepseek-qwen-aggressive-37task.json`
  - `research/evals/20260628-probe-gated-latency-calibration-triple-aggressive-37task.json`
  - `research/evals/20260628-oracle-latency-calibration-37task.json`

Selected policy:

- source latency cost: `0.5` per second
- min probability ratio: `0.0`
- min probability: `0.0`

Results on 37 measured rows:

| policy | calibrated rows | pass@1 | solvable pass@1 | target accuracy | mean latency | latency regret | changed routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base logits router | 0 | 0.8108 | 0.9375 | 0.7027 | 5593.8 ms | 1469.5 ms | 0 |
| DeepSeek + Qwen probe gate | 18 | 0.8108 | 0.9375 | 0.8919 | 5009.9 ms | 1127.2 ms | 7 |
| DeepSeek + Kimi + Qwen probe gate | 16 | 0.8108 | 0.9375 | 0.8919 | 5009.9 ms | 1127.2 ms | 7 |
| oracle latency-safe gate | 16 | 0.8108 | 0.9375 | 0.8919 | 5009.9 ms | 1127.2 ms | 7 |

Learning:

Probe-gated latency calibration preserves pass@1 while improving target
accuracy and latency regret on the 37-task measured slice. The triple-probe
gate matches the oracle latency-safe gate exactly. The cheaper DeepSeek + Qwen
gate includes two extra probe-positive rows, but the calibration utility does
not change those rows into harmful routes, so it lands on the same selected
policy.

This is the first deployable-shaped conditional calibration result: the gate is
derived from observed probe outcomes rather than the oracle all-pass label.

Decision:

Keep this path. The next policy artifact should include a probe-gated
calibration configuration rather than only an oracle task-id allowlist.

Next step:

Promote the probe-gated evaluation into a reusable policy artifact and compare
it in the normal baseline report beside strongest single worker, fastest worker,
active logits router, and oracle calibration.
