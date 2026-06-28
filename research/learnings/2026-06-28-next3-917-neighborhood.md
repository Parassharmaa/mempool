# Next3 917 Neighborhood

Run tag: `20260628-next3-917-neighborhood`

## Question

Does the newly found `BigCodeBench/917` Kimi target sit inside a broader
Kimi/GLM-positive neighborhood that can safely improve the active router?

## Planner Fix

The positive-neighborhood planner initially failed when excluding historical
screen summaries because one older summary used a legacy `by_task` mapping
instead of the newer list-of-task records shape. The helper now accepts both:

- `tools/plan_solvability_aware_specialist_acquisition.py`
- `tests/test_plan_solvability_aware_specialist_acquisition.py`

This keeps acquisition planning compatible with older run artifacts.

## Candidate Selection

The planner used the 26-task candidate context that includes `917` as a Kimi
positive seed:

- routing context:
  `research/datasets/20260628-mixed-winner-24task-next3-routing.jsonl`
- task source:
  `research/evals/bigcodebench_hard_next3_profile_eligible_merged_tasks.json`
- planned tasks:
  `research/evals/20260628-next3-917-neighborhood-tasks.json`
- report:
  `research/evals/20260628-next3-917-neighborhood-report.json`

Selected tasks:

- Kimi-neighborhood: `BigCodeBench/1057`, `BigCodeBench/92`
- GLM-neighborhood: `BigCodeBench/486`, `BigCodeBench/184`

## Specialist-First Screen

The non-Qwen screen ran GLM, DeepSeek, and Kimi once per selected task:

- outcomes:
  `research/evals/results/20260628-next3-917-neighborhood-screen1.jsonl`
- summary:
  `research/evals/results/20260628-next3-917-neighborhood-screen1-summary.json`

Results:

- `92`: passed by GLM, DeepSeek, and Kimi
- `184`: passed by GLM, DeepSeek, and Kimi
- `1057`: universal failure
- `486`: universal failure

## Top4 Repeat

The positive tasks `92` and `184` were repeated across the top-four pool with
two samples per worker:

- outcomes:
  `research/evals/results/20260628-next3-917-neighborhood-positive-repeat.jsonl`
- summary:
  `research/evals/results/20260628-next3-917-neighborhood-positive-repeat-summary.json`
- routing rows:
  `research/datasets/20260628-next3-917-neighborhood-positive-repeat-routing.jsonl`

Both tasks were broad-pass rows. Every top-four worker passed both samples, but
Qwen was much faster and became the latency-adjusted target:

- `92`: Qwen 2/2, mean latency 3243 ms
- `184`: Qwen 2/2, mean latency 4495.5 ms

The active router already predicts Qwen for both rows:

- `research/evals/results/20260628-next3-active-policy-on-917-neighborhood.json`

## Refresh Attempt

The run tested whether adding `917` plus the two Qwen latency anchors could
stabilize a larger refresh:

- merged dataset:
  `research/datasets/20260628-mixed-winner-26task-next3-917-neighborhood-routing.jsonl`
- temperature selection:
  `research/datasets/20260628-mixed-winner-26task-next3-917-neighborhood-temperature-selection.json`
- refresh decision:
  `research/policies/20260628-mixed-winner-26task-next3-917-neighborhood-refresh.json`

Decision: quarantine. The best candidate was temperature `0.05`, with:

- LOO target accuracy: 0.692
- LOO solvable pass@1: 0.739
- LOO mean latency regret: 713.2 ms

It missed the minimum target accuracy and exceeded the allowed latency-regret
increase over the active 23-task policy.

## Decision

Keep the new rows and the planner compatibility fix, but do not promote a new
active router.

The important learning is that `917` is a sharp exception in a nearby
scientific/data-analysis region: Qwen should remain the target for broad-pass
neighbors like `92` and `184`, but Qwen fails `917` while Kimi and GLM pass.
The active policy needs more examples that distinguish Qwen-fast broad-pass rows
from Qwen-fail Kimi/GLM rows before another refresh can pass the gate.
