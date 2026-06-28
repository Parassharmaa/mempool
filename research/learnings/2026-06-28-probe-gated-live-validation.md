# Probe-Gated Live Validation Attempt

Ran a small live cloud validation batch for the frozen probe-gated policy.

Frozen policy:

- `research/policies/20260628-probe-gated-latency-calibrated-policy.json`

Selected tasks:

- `research/evals/20260628-probe-gated-live-validation-tasks.json`
- selection report:
  `research/evals/20260628-probe-gated-live-validation-selection.json`

Live run:

- config:
  `research/evals/ollama_cloud_worker_pool_top4.json`
- outcomes:
  `research/evals/results/20260628-probe-gated-live-validation-top4.jsonl`
- summary:
  `research/evals/results/20260628-probe-gated-live-validation-top4-summary.json`
- routing dataset:
  `research/datasets/20260628-probe-gated-live-validation-top4-routing.jsonl`
- baseline report:
  `research/evals/20260628-probe-gated-live-validation-baseline-report.json`

Result:

- 4 tasks
- 4 workers
- 2 samples per worker/task
- 32 outcome rows
- 0 passing samples

Baseline report:

| policy | pass@1 | target accuracy | mean latency | latency regret | calibrated rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| strongest worker | 0.0000 | 0.2500 | 7420.2 ms | 3446.0 ms | n/a |
| fastest worker | 0.0000 | 0.2500 | 7420.2 ms | 3446.0 ms | n/a |
| active logits router | 0.0000 | 0.2500 | 14278.2 ms | 10304.0 ms | n/a |
| probe-gated calibrated router | 0.0000 | 0.2500 | 14278.2 ms | 10304.0 ms | 0 |
| oracle target | 0.0000 | 1.0000 | 3974.2 ms | 0.0 ms | n/a |

Learning:

This live batch is not useful as a positive validation of the probe-gated
policy because every worker failed every sample. The policy did behave safely:
the probe gate calibrated zero rows, so it did not introduce harmful latency
changes on universal failures.

The selection heuristic was too optimistic. Low environment risk and task
novelty are not enough for live validation; the next live policy validation
batch should first pass a cheap solvability screen, such as one Qwen sample or a
canonical-solution environment check, before spending repeated top-4 calls.

Decision:

Discard this as policy-validation evidence. Keep it as a negative-control
selection lesson.

Next step:

Build a solvability-screened live validation batch:

1. select fresh tasks outside the policy-source and replay-heldout slices,
2. run one cheap Qwen sample per task,
3. retain only tasks with at least one pass or a verified canonical environment
   pass,
4. then run the frozen top-4 repeated comparison.
