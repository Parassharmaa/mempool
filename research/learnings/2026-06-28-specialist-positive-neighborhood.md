# Specialist Positive Neighborhood

Run tag: `20260628-specialist-positive-neighborhood`

Goal: build a stronger specialist acquisition selector by starting from tasks
that the target worker actually passed, then selecting fresh tasks near those
worker-positive neighborhoods.

Change:

Added `tools/plan_specialist_positive_neighborhood.py`.

The selector:

- builds positive seeds per target worker from routing records where that worker
  passed
- ranks fresh BigCodeBench tasks by similarity to that target worker's positive
  seeds
- excludes routing records and prior screening summaries, including older
  summary formats that only contain `records`
- emits a manifest compatible with target-specialist screening

Generated plan:

- DeepSeek candidates: `1057`, `800`
- GLM candidates: `486`, `399`
- Kimi candidates: `560`, `765`

Seed counts:

- DeepSeek: `38`
- GLM: `34`
- Kimi: `38`

Bounded screen:

Ran one Kimi sample each on `BigCodeBench/560` and `BigCodeBench/765`.

Result:

- Kimi: `0/2`
- Graduated tasks: `0`

Decision: keep the target-specific positive-neighborhood selector, but do not
spend more calls on this candidate source without a stronger mining gate.

Learning:

The selector is better aligned than broad solvability priors because it uses
target-worker positive evidence, but it still selected tasks that Kimi could not
solve on first sample. The next acquisition step should mine actual new
single-worker positives first, then compare those positives across the pool.
Fresh similarity selection alone is still too noisy for the current benchmark
slice.

Next step:

Run a cheap specialist mining pass over a fresh low-risk task pool for one
worker at a time, then only graduate passing tasks to repeated top-4 comparison.

Artifacts:

- `tools/plan_specialist_positive_neighborhood.py`
- `research/programs/20260628-specialist-positive-neighborhood-plan.json`
- `research/evals/20260628-specialist-positive-neighborhood-tasks.json`
- `research/programs/20260628-positive-neighborhood-kimi-screen1.json`
- `research/evals/results/20260628-positive-neighborhood-kimi-screen1-summary.json`
- `research/evals/20260628-positive-neighborhood-kimi-screen1-graduated-tasks.json`
