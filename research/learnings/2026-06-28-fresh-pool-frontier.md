# Fresh Pool Frontier

Run tag: `20260628-fresh-pool-frontier`

## Question

Can the acquisition selector find truly fresh tasks after the current BigCodeBench-Hard eligible pool has been consumed by prior outcome runs?

## Result

Keep as a sourcing milestone.

- The BigCodeBench-Hard frontier scan at offset `148` returned `0` rows scanned and `0` eligible tasks, confirming the current hard split frontier is exhausted for this loader configuration.
- The existing hard candidate reports remain useful for held-out analysis, but strict exclusion against `research/evals/results/` leaves no unseen hard acquisition candidates.
- A fallback scan over `bigcode/bigcodebench` split `v0.1.4` from offset `0` scanned `16` rows and found `12` locally eligible instruct-mode tasks.
- The value-head acquisition selector, with strict routing and outcome exclusions, found `10` candidates and selected `6` tasks:
  - `bigcodebench-hard-BigCodeBench-12`
  - `bigcodebench-hard-BigCodeBench-6`
  - `bigcodebench-hard-BigCodeBench-14`
  - `bigcodebench-hard-BigCodeBench-7`
  - `bigcodebench-hard-BigCodeBench-1`
  - `bigcodebench-hard-BigCodeBench-4`

## Interpretation

This is not a new hard-benchmark score. It is a training-data expansion path for the orchestrator and second-attempt value head. The local loader preserves the `bigcodebench-hard-BigCodeBench-*` task-id prefix for these materialized tasks, so source provenance must come from the scan report, not task-id text alone.

The selected batch is useful because it concentrates on uncertain value-head regions:

- filesystem/subprocess tasks where fallback value may be real but costly;
- small top-vs-second margins where the router is uncertain;
- cases similar to prior false-spend or missed-positive examples.

## Artifacts

- Hard frontier exhaustion report: `research/evals/20260628-fresh-pool-frontier-offset148-report.json`
- Normal BigCodeBench eligible pool: `research/evals/20260628-fresh-pool-frontier-bigcodebench-offset0-tasks.json`
- Normal BigCodeBench scan report: `research/evals/20260628-fresh-pool-frontier-bigcodebench-offset0-report.json`
- Selected acquisition batch: `research/evals/20260628-fresh-pool-frontier-value-head-tasks.json`
- Selection report: `research/evals/20260628-fresh-pool-frontier-value-head-report.json`

## Next Step

Run a bounded real-worker outcome pass on the six selected acquisition tasks, then fold the results into the outcome ledger before retraining or changing the active policy.
