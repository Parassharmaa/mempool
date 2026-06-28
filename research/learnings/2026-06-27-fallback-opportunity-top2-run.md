# Fallback Opportunity Top-2 Run

## Question

Do the first two selected fallback-opportunity tasks produce useful second-worker
rescue labels for the fallback head?

## Run

Ran the first two tasks from
`research/evals/bigcodebench_hard_fallback_opportunity_batch8_tasks.json`
against the top-4 cloud worker pool:

- GLM 5.2
- DeepSeek V4 Pro
- Kimi K2.7 Code
- Qwen3 Coder 480B

Each worker was sampled twice per task.

Artifacts:

- `research/evals/results/20260627-fallback-opportunity-top2-repeat.json`
- `research/evals/results/20260627-fallback-opportunity-top2-repeat.jsonl`
- `research/evals/results/20260627-fallback-opportunity-top2-repeat-summary.json`
- `research/datasets/20260627-fallback-opportunity-top2-repeat-routing.jsonl`
- `research/datasets/20260627-fallback-opportunity-top2-fallback-training.jsonl`
- `research/datasets/20260627-fallback-opportunity-top2-fallback-training-report.json`

## Result

Both tasks were universal failures across all workers and both samples:

- `BigCodeBench/771`: 0/8 successful samples
- `BigCodeBench/785`: 0/8 successful samples

The fallback-specific training report contains:

- task count: 2
- fallback opportunities: 2
- useful fallbacks: 0
- fallback hurts: 2
- solvable tasks: 0

The routing conversion picked GLM as target on both rows only because it was the
fastest failing worker. These rows should not be treated as positive routing
evidence.

## Learning

Low top-two router margin is not enough to find useful fallback labels. The
selector found uncertainty, but the first two tasks were too hard for every
worker. For fallback-head data acquisition, we need a solvability prior:

- mine tasks where at least one worker has already shown a pass nearby, or
- run a cheap single-sample solvability screen before spending repeated samples,
  or
- seed from known top-fail/second-pass examples rather than generic low-margin
  uncertainty alone.

The new fallback-specific dataset builder is useful because it keeps these rows
as explicit negative action labels instead of letting all-fail latency targets
pollute the router training set.

## Next Step

Add a screened acquisition stage: run one sample from the current top worker and
one specialist on a larger candidate list, keep only rows where at least one
worker passes or where the top worker fails with a plausible second-worker pass,
then spend repeated samples.
