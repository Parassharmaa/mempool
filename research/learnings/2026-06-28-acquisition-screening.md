# Acquisition Screening

Run tag: `20260628-acquisition-screening`

Goal: prevent repeated acquisition runs from silently feeding universal-failure
tasks into the routing dataset.

Change:

Extended `tools/summarize_repeated_outcomes.py` with task-level acquisition
decisions:

- `candidate_task_ids`: tasks with at least one passing worker sample
- `universal_failure_task_ids`: tasks where every worker/sample failed
- per-task best empirical worker, pass rate, and latency when a task has any
  passing sample

Result:

The bounded specialist acquisition smoke for `BigCodeBench/208` now summarizes
as:

- `candidate_task_ids`: empty
- `universal_failure_task_ids`: `bigcodebench-hard-BigCodeBench-208`
- `candidate_for_conversion`: `false`

Decision: keep.

This makes the data-acquisition loop safer: fresh measured rows can be screened
before conversion, and universal failures can be used as harness/task-selection
evidence without becoming misleading router labels.

Next step:

Run the remaining specialist acquisition tasks in small batches, summarize each
batch, and only convert tasks with at least one confirmed passing worker. For
tasks with multiple passing workers, choose routing targets from empirical
pass-rate first and latency/reward second.

Artifacts:

- `tools/summarize_repeated_outcomes.py`
- `research/evals/results/20260628-specialist-acquisition-limit1-summary.json`
