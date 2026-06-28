# Probe-Gated Baseline Report

Added probe-gated latency calibration to the standard router baseline report.

Implementation:

- `tools/train_router_baseline.py`
  - accepts `--probe-gated-policy`
  - loads a reusable probe-gated calibration artifact
  - reports the policy beside family, nearest-neighbor, strongest-worker,
    fastest-worker, oracle, and active logits-router baselines
- `research/policies/20260628-probe-gated-latency-calibrated-policy.json`
  - stores the base logits-router model
  - stores the DeepSeek + Qwen probe gate
  - stores the selected latency calibration settings
- `tests/test_router_baseline.py`

Experiment:

- dataset:
  `research/datasets/20260628-latency-safe-head-37task-routing.jsonl`
- report:
  `research/evals/20260628-probe-gated-baseline-report-37task.json`

Comparison:

| policy | pass@1 | solvable pass@1 | target accuracy | mean latency | latency regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| strongest worker | 0.7297 | 0.8438 | 0.1351 | 10162.9 ms | 6151.8 ms |
| fastest worker | 0.6757 | 0.7812 | 0.6216 | 4009.9 ms | 627.4 ms |
| active logits router | 0.8108 | 0.9375 | 0.7027 | 5593.8 ms | 1469.5 ms |
| probe-gated calibrated router | 0.8108 | 0.9375 | 0.8919 | 5009.9 ms | 1127.2 ms |
| oracle target | 0.8649 | 1.0000 | 1.0000 | 4124.4 ms | 0.0 ms |

Learning:

The probe-gated policy is now comparable in the same artifact as the normal
baselines. On this measured slice, it improves target accuracy and latency
regret over the active logits router without hurting pass@1. It does not beat
the fastest worker on raw latency, but it keeps substantially higher pass@1.

The report also highlights a standing caveat: in-sample nearest-neighbor can
memorize task labels, so it is useful as an upper diagnostic only when evaluated
on the same routing dataset. The meaningful learned-policy comparison is the
active logits router versus the probe-gated calibrated router.

Decision:

Keep the baseline-report integration and the policy artifact.

Next step:

Run a fresh held-out measured batch and evaluate the saved probe-gated policy
without retuning its probe gate or calibration constants.
