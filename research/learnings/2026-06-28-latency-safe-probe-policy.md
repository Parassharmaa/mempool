# Latency-Safe Probe Policy

Tested whether cheap measured probe outcomes can identify rows where latency
optimization is safe.

Implementation:

- `src/mempool/latency_safe_probe.py`
  - evaluates one or more probe workers against the latency-safe label
  - supports `all` and `any` probe agreement modes
  - records confusion metrics and observed probe latency
- `tools/evaluate_latency_safe_probe_policy.py`
  - sweeps probe-worker combinations over a routing dataset
- `tests/test_latency_safe_probe.py`

Experiment:

- dataset:
  `research/datasets/20260628-latency-safe-head-37task-routing.jsonl`
- report:
  `research/evals/20260628-latency-safe-probe-policy-37task.json`

Best probe policies on the 37-task measured dataset:

| probe policy | mode | precision | recall | accuracy | mean probe latency |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek + Kimi + Qwen | all | 1.0000 | 1.0000 | 1.0000 | 31423.5 ms |
| DeepSeek + Kimi | all | 0.9412 | 1.0000 | 0.9730 | 27413.7 ms |
| DeepSeek + Qwen | all | 0.8889 | 1.0000 | 0.9459 | 14172.8 ms |
| GLM | all | 0.8000 | 1.0000 | 0.8919 | 20741.3 ms |

Comparison against static heads:

| method | best precision | recall | note |
| --- | ---: | ---: | --- |
| prompt + router-confidence head | 0.6000 | 0.3750 | leave-one-out |
| + historical reliability features | 0.6000 | 0.3750 | leave-one-out |
| GLM single probe | 0.8000 | 1.0000 | measured probe |
| best triple probe | 1.0000 | 1.0000 | measured probe |

Learning:

Probe outcomes are much stronger than static prompt/router features for the
latency-safe condition on this measured slice. A single GLM probe already beats
the learned latency-safe head, and multi-probe agreement can nearly eliminate
false positives in the current dataset.

The cost tradeoff is real. The perfect policy uses three worker calls and should
not be treated as a default runtime path. A cheaper pair such as DeepSeek + Qwen
is a more practical candidate when the expected latency-calibration gain is
large enough to pay for the probe.

Decision:

Keep the probe-policy machinery. It is the first non-oracle condition that
looks strong enough to gate conditional latency calibration.

Next step:

Integrate the best practical probe policy into the conditional calibration
evaluator as a runtime gate, then compare:

1. base logits router,
2. oracle latency-safe calibration,
3. probe-gated latency calibration,
4. strongest single worker baseline.
