# Solvability-Aware Specialist Plan

Run tag: `20260628-solvability-aware-specialist-plan`

Goal: replace the weak specialist-acquisition queue with fresh candidates that
use a solvability prior before specialist similarity.

Change:

Added a solvability-aware specialist acquisition planner. It builds a simple
positive prior from existing routing records:

- categories seen in passing records
- libraries seen in passing records
- nearest positive task-shape similarity
- environment-risk and plausibility penalties

The selector then combines that solvability score with specialist-miss
similarity, while excluding existing routing records and prior screening
summaries.

Result:

The regenerated plan excludes the ten tasks already screened or full-compared
and selects nine fresh candidates:

- DeepSeek: `124`, `560`, `486`
- GLM: `399`, `509`, `458`
- Kimi: `579`, `771`, `15`

The plan uses `47` positive prior records from the current 50-task routing
dataset.

Bounded target screen:

Ran Kimi on the first two fresh Kimi candidates:

- `BigCodeBench/579`: fail
- `BigCodeBench/771`: fail

Graduated tasks: `0`

Decision: keep the selector, but do not treat the current scoring formula as
sufficient for data acquisition by itself.

The fresh selector improved hygiene by avoiding known dead tasks, but it still
did not produce a passing specialist row in the first two-call screen. The next
selector should use stronger evidence than broad category/library priors:
canonical-pass confidence, direct historical positive neighborhoods, or actual
single-worker mining results should be required before specialist screening.

Next step:

Build a "positive-neighborhood first" selector from already measured passing
tasks, then choose nearby fresh tasks for the intended specialist. Use the
target-specialist screen as the first gate and only full-compare rows that pass
that gate.

Artifacts:

- `src/mempool/acquisition_plan.py`
- `tools/plan_solvability_aware_specialist_acquisition.py`
- `research/programs/20260628-solvability-aware-specialist-acquisition-plan.json`
- `research/evals/20260628-solvability-aware-specialist-acquisition-tasks.json`
- `research/programs/20260628-solvability-aware-kimi-screen1.json`
- `research/evals/results/20260628-solvability-aware-kimi-screen1-summary.json`
- `research/evals/20260628-solvability-aware-kimi-screen1-graduated-tasks.json`
