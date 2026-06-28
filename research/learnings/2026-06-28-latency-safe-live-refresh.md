# Latency-Safe Live Refresh

Ran the matched-control top-4 worker batch:

- manifest:
  `research/evals/20260628-latency-safe-matched-controls-manifest.json`
- outcomes:
  `research/evals/results/20260628-latency-safe-matched-controls-top4.jsonl`
- summary:
  `research/evals/results/20260628-latency-safe-matched-controls-top4-summary.json`
- routing dataset:
  `research/datasets/20260628-latency-safe-matched-controls-top4-routing.jsonl`
- expanded latency-safe dataset:
  `research/datasets/20260628-latency-safe-head-37task-routing.jsonl`

Live outcome shape:

- 64 outcome rows
- 8 tasks
- 4 workers
- 2 samples per worker/task
- new latency-safe rows: 3
- new unsafe rows: 5

New task labels:

- latency-safe:
  - `bigcodebench-hard-BigCodeBench-127`
  - `bigcodebench-hard-BigCodeBench-384`
  - `bigcodebench-hard-BigCodeBench-547`
- unsafe:
  - `bigcodebench-hard-BigCodeBench-305`
  - `bigcodebench-hard-BigCodeBench-320`
  - `bigcodebench-hard-BigCodeBench-325`
  - `bigcodebench-hard-BigCodeBench-336`
  - `bigcodebench-hard-BigCodeBench-391`

Retrained latency-safe head:

- report:
  `research/evals/20260628-latency-safe-head-37task-report.json`
- sweep:
  `research/evals/20260628-latency-safe-head-37task-sweep.json`
- model:
  `research/models/20260628-latency-safe-head-37task.json`

Result:

| dataset | best LOO precision | best LOO accuracy | best LOO recall |
| --- | ---: | ---: | ---: |
| previous 29 rows | 0.6250 | 0.6207 | 0.3846 |
| expanded 37 rows | 0.6000 | 0.6216 | 0.3750 |

Learning:

The matched-control acquisition produced useful labels, but the current
latency-safe logit head did not improve. In-sample performance is still high,
while leave-one-out precision remains too low for runtime use. This suggests
the present prompt/router-confidence features are not enough to distinguish
all-pass latency-safe tasks from nearby unsafe controls.

Next step:

Stop threshold tuning this head. The next useful experiment should add a
stronger condition source, such as:

1. cheap probe outcomes from one fast worker before applying latency
   calibration,
2. richer features from dependency/category neighborhoods and historical
   per-worker reliability,
3. a verifier-style head trained on worker agreement rather than only prompt
   features.

Keep conditional latency calibration as an oracle diagnostic until a higher
precision condition is available.
