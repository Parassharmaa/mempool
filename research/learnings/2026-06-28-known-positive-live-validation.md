# Known-Positive Live Validation

Validated the frozen probe-gated policy on a live known-positive slice.

Candidate source:

- `research/evals/20260628-known-positive-live-screen-candidates12-tasks.json`
- `research/evals/20260628-known-positive-live-screen-candidates12-selection.json`

Screen:

- top-4 workers
- 1 sample per worker/task
- outcomes:
  `research/evals/results/20260628-known-positive-live-top4-screen1.jsonl`
- positives:
  `research/evals/20260628-known-positive-live-top4-screen1-positive-tasks.json`

Repeated validation:

- 7 graduated tasks
- top-4 workers
- 2 samples per worker/task
- outcomes:
  `research/evals/results/20260628-known-positive-live-top4-repeat2.jsonl`
- routing dataset:
  `research/datasets/20260628-known-positive-live-top4-repeat2-routing.jsonl`
- baseline report:
  `research/evals/20260628-known-positive-live-probe-gated-baseline-report.json`

Result:

| policy | pass@1 | target accuracy | mean latency | latency regret |
| --- | ---: | ---: | ---: | ---: |
| strongest worker | 1.0000 | 0.5714 | 6272.0 ms | 1500.0 ms |
| fastest worker | 1.0000 | 0.5714 | 6272.0 ms | 1500.0 ms |
| active logits router | 0.7143 | 0.2857 | 8188.3 ms | 3416.3 ms |
| probe-gated calibrated router | 0.7143 | 0.5714 | 5702.3 ms | 1077.4 ms |
| oracle target | 1.0000 | 1.0000 | 4772.0 ms | 0.0 ms |

Slice shape:

- 56 live outcome rows
- 37 passing samples
- 7 tasks
- 3 all-pass latency-safe rows
- probe-gated policy calibrated 4 rows and changed 2 routes

Learning:

The frozen probe-gated policy generalized in the narrow sense we needed: it
improved the active logits router's target accuracy and latency regret without
changing active-router pass@1. It did not beat the strongest/fastest
single-worker baselines on this known-positive slice; those baselines solved all
7 tasks.

This means the conditional calibration mechanism is useful, but the underlying
active logits router remains the bottleneck. The next router refresh should
include this live repeated dataset and should be gated against strongest-worker
pass@1, not only target accuracy and latency regret.

Decision:

Keep this as live validation evidence for the probe-gated mechanism, with the
explicit caveat that the current active router is not yet competitive with the
best single-worker baseline on this slice.

Next step:

Merge this live repeated routing dataset into the refresh candidate pool and
train/evaluate a new router candidate with a pass@1-preserving gate against the
strongest-worker baseline.
