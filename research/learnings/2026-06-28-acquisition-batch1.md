# Specialist Acquisition Batch 1

Run tag: `20260628-acquisition-batch1`

Goal: continue specialist data acquisition in small screened batches, skipping
known universal failures before spending more worker calls.

Change:

Added `tools/plan_acquisition_batch.py`, a manifest-driven batch planner that:

- reads the specialist acquisition manifest
- reads one or more screening summaries
- skips already-screened and universal-failure tasks
- emits a bounded task file and batch manifest
- records the exact benchmark command and follow-up summary command
- can filter by `environment_risk` from the acquisition manifest metadata

Batch results:

High-risk source-order batch:

- Tasks: `BigCodeBench/942`, `BigCodeBench/636`
- Calls: `16`
- Passes: `0`
- Failure modes: `16` `test_failure`
- Screening decision: both tasks are universal failures

Low-risk batch:

- Tasks: `BigCodeBench/985`, `BigCodeBench/988`
- Calls: `16`
- Passes: `0`
- Failure modes: `15` `test_failure`, `1` `request_timeout`
- Screening decision: both tasks are universal failures

Current acquisition evidence:

Across the first five acquisition tasks tested so far (`208`, `942`, `636`,
`985`, `988`), every task is a universal failure under the current four-worker
pool. Lowering environment risk from plotting/data-science tasks to
filesystem/data tasks did not produce candidate routing rows.

Decision: keep the batch planner, but stop expanding this specialist-acquisition
branch through full four-worker repeated comparisons until a cheaper
solvability screen is added.

Next step:

Add a pre-screen stage that runs one strong/fast incumbent worker on candidate
tasks with one sample. Only tasks with at least one pass should graduate to the
full repeated four-worker comparison. This should reduce wasted calls and focus
the orchestrator refresh on rows that can actually become routing labels.

Artifacts:

- `tools/plan_acquisition_batch.py`
- `research/programs/20260628-specialist-acquisition-batch1.json`
- `research/programs/20260628-specialist-acquisition-lowrisk-batch1.json`
- `research/evals/20260628-specialist-acquisition-batch1-tasks.json`
- `research/evals/20260628-specialist-acquisition-lowrisk-batch1-tasks.json`
- `research/evals/results/20260628-specialist-acquisition-batch1-summary.json`
- `research/evals/results/20260628-specialist-acquisition-lowrisk-batch1-summary.json`
