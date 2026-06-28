# Positive-Neighborhood Fallback Screen

## Question

Can fallback-opportunity acquisition improve by selecting tasks near known
solvable BigCodeBench rows instead of relying on router uncertainty alone?

## Change

Added `tools/select_positive_neighborhood_fallback_batch.py`.

The selector uses known-positive routing datasets as seeds, ranks fresh
candidate tasks by similarity to those positives, and still records active-router
top/second worker probabilities so the output can feed fallback screening.

Generated:

- `research/evals/bigcodebench_hard_positive_neighborhood_fallback6_tasks.json`
- `research/evals/bigcodebench_hard_positive_neighborhood_fallback6_report.json`

## Screen Run

Ran a bounded one-sample screen on the top two selected tasks:

- `BigCodeBench/594`
- `BigCodeBench/720`

Artifacts:

- `research/evals/results/20260627-positive-neighborhood-screen2.json`
- `research/evals/results/20260627-positive-neighborhood-screen2.jsonl`
- `research/evals/results/20260627-positive-neighborhood-screen2-summary.json`
- `research/datasets/20260627-positive-neighborhood-screen2-routing.jsonl`
- `research/datasets/20260627-positive-neighborhood-screen2-fallback-training.jsonl`
- `research/datasets/20260627-positive-neighborhood-screen2-fallback-training-report.json`

## Result

Both screened tasks were universal failures across the top-4 cloud worker pool:

- GLM 5.2: 0/2
- DeepSeek V4 Pro: 0/2
- Kimi K2.7 Code: 0/2
- Qwen3 Coder 480B: 0/2

The fallback-action report contains:

- task count: 2
- fallback opportunities: 2
- useful fallbacks: 0
- fallback hurts: 2
- solvable tasks: 0

Across the fallback acquisition attempts so far, we now have 8 fallback
opportunities and 0 useful fallback positives from the candidate-screening path.

## Learning

Positive-neighborhood similarity is better structured than uncertainty-only
selection, but it still does not guarantee solvability. For this benchmark
slice, many tasks near solvable examples remain too brittle under single-sample
cloud generation.

The next acquisition rule should require a stronger solvability prerequisite
before fallback screening:

- canonical-pass in the local harness is necessary but not sufficient;
- at least one actual worker pass in a cheap mining pass is a better gate;
- fallback positives should be mined from known worker-positive tasks by
  comparing first-choice router predictions against all worker outcomes, rather
  than asking fresh uncertain tasks to produce rescues.

## Next Step

Build a mining tool over existing outcome/routing datasets that extracts
historical top-fail/alternate-pass cases under the active router. This can yield
fallback-head training positives without additional cloud calls, and it can tell
us which task neighborhoods deserve fresh spending.
