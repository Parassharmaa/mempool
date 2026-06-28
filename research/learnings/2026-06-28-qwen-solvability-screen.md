# Qwen Solvability Screen

Run tag: `20260628-qwen-solvability-screen`

Goal: add a cheap pre-screen stage before full repeated specialist comparisons,
so acquisition candidates must show at least one useful pass signal before they
graduate into the expensive four-worker comparison loop.

Change:

Extended `tools/plan_acquisition_batch.py` so batch manifests can override:

- worker pool
- repeat count
- benchmark id
- global-universal threshold via `--universal-min-workers`

This lets the same planner emit a one-worker, one-sample solvability screen
without confusing it with a full four-worker repeated comparison.

Important guardrail:

A one-worker screen failure is not a global universal failure. The planner now
only treats a task as globally universal when the task-level summary has at
least the configured worker count. For this project, full-pool universal
failures use `--universal-min-workers 4`.

Screen result:

Ran Qwen3 Coder 480B once on four unscreened acquisition tasks:

- `BigCodeBench/513`
- `BigCodeBench/528`
- `BigCodeBench/857`
- `BigCodeBench/211`

All four failed with `test_failure`.

Graduation result:

`research/evals/20260628-specialist-acquisition-qwen-screen1-graduated-tasks.json`
contains zero tasks, so no tasks from this incumbent screen should graduate to a
full repeated comparison.

Decision: keep the pre-screen machinery.

The result confirms the value of a cheap screen: it ruled out four incumbent
passes using four calls instead of a 32-call repeated four-worker batch. But it
does not prove these tasks are impossible specialist wins. A future specialist
screen should run the intended target worker for each remaining target cluster,
especially for Kimi/GLM/DeepSeek-positive discovery.

Next step:

Add a target-worker pre-screen for specialist acquisition: one sample from the
intended specialist per candidate task, then graduate only specialist-passing
tasks to the full comparison with Qwen included.

Artifacts:

- `tools/plan_acquisition_batch.py`
- `research/programs/20260628-specialist-acquisition-qwen-screen1.json`
- `research/evals/20260628-specialist-acquisition-qwen-screen1-tasks.json`
- `research/evals/results/20260628-specialist-acquisition-qwen-screen1.jsonl`
- `research/evals/results/20260628-specialist-acquisition-qwen-screen1-summary.json`
- `research/evals/20260628-specialist-acquisition-qwen-screen1-graduated-tasks.json`
