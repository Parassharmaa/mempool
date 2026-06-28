# Value-Head Acquisition Selector

Run tag: `20260628-value-head-acquisition-selector`

## Question

Can the corrected held-out second-attempt value-head errors drive the next
BigCodeBench acquisition batch?

## Change

Added `tools/select_value_head_acquisition_batch.py`, which:

- reads the held-out value-head report;
- extracts false negatives where the oracle value label was positive but the
  learned head did not spend a fallback;
- extracts false positives where the learned head spent a fallback on a
  non-positive value label;
- builds seed analyses from the small-orchestrator substrate and multi-head LOO
  worker distributions;
- scores candidate BigCodeBench tasks by seed similarity, active-router
  uncertainty, likely second worker, environment risk, and plausibility;
- writes both a task file and a ranked report.

The tool also supports `--exclude-outcome-dir`, so future acquisition commands
can exclude every prior JSONL outcome row without hand-listing files.

## Result

Strict fresh mode:

- task sources: next3/top4 eligible merged pools;
- excluded prior routing datasets and every JSONL outcome in
  `research/evals/results`;
- candidate count: `0`;
- selected tasks: none.

Relaxed diagnostic mode:

- candidate count: `66`;
- selected diagnostic ranking:
  - `bigcodebench-hard-BigCodeBench-1015`
  - `bigcodebench-hard-BigCodeBench-346`
  - `bigcodebench-hard-BigCodeBench-15`
  - `bigcodebench-hard-BigCodeBench-594`
  - `bigcodebench-hard-BigCodeBench-399`
  - `bigcodebench-hard-BigCodeBench-579`

Several relaxed selections were already known from earlier outcome rows, which
confirmed that strict outcome-ledger exclusion is necessary.

## Interpretation

The selector is useful, but the current eligible candidate pool is exhausted
once prior outcome rows are respected. The next action should not be another
selection pass over the same merged pool. It should materialize or unlock fresh
BigCodeBench tasks first, then run this selector against that new pool.

This is aligned with the orchestration goal because the held-out value head needs
more positive and negative workflow-action labels before threshold tuning or
architecture promotion is meaningful.

## Artifacts

- `tools/select_value_head_acquisition_batch.py`
- `tests/test_select_value_head_acquisition_batch.py`
- `research/evals/20260628-value-head-acquisition-selector-report.json`
- `research/evals/20260628-value-head-acquisition-selector-tasks.json`
- `research/evals/20260628-value-head-acquisition-selector-relaxed-report.json`
- `research/evals/20260628-value-head-acquisition-selector-relaxed-tasks.json`
