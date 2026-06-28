# Fallback Error Neighborhood Screen

Run tag: `20260627-fallback-error-neighborhood-screen4`

## What Ran

Screened four targeted BigCodeBench tasks selected from the mined fallback
head's leave-one-out error neighborhoods:

- missed-positive side: `BigCodeBench/988`, `BigCodeBench/765`
- false-positive side: `BigCodeBench/492`, `BigCodeBench/579`

Each task was evaluated once across the top-4 Ollama Cloud worker pool:

- GLM 5.2
- DeepSeek V4 Pro
- Kimi K2.7 Code
- Qwen3 Coder 480B

Artifacts:

- task file: `research/evals/bigcodebench_hard_fallback_error_neighborhood_screen4_tasks.json`
- outcomes: `research/evals/results/20260627-fallback-error-neighborhood-screen4.jsonl`
- summary: `research/evals/results/20260627-fallback-error-neighborhood-screen4-summary.json`
- routing dataset: `research/datasets/20260627-fallback-error-neighborhood-screen4-routing.jsonl`
- mined cases: `research/datasets/20260627-fallback-error-neighborhood-screen4-mined-cases.jsonl`
- retrained head: `research/models/20260627-mined-fallback-head-plus-error-neighborhood.json`

## Result

All 16 worker-task outcomes failed.

Worker means:

- GLM 5.2: 0/4, mean latency 10285.25 ms
- DeepSeek V4 Pro: 0/4, mean latency 8980.75 ms
- Kimi K2.7 Code: 0/4, mean latency 16921.75 ms
- Qwen3 Coder 480B: 0/4, mean latency 3654.25 ms

The active router saw all four rows as fallback opportunities, but none were
solvable by any worker:

- fallback opportunities: 4
- useful fallbacks: 0
- hard negatives: 4
- solvable fallback opportunities: 0

## Training Effect

Adding these four hard negatives to the mined fallback-action dataset changed
the deduplicated set from 16 tasks to 20 tasks:

- positives: still 3
- negatives: 17

The mined fallback head still fits training perfectly, but leave-one-out changed
from:

- previous LOO F1: 0.333
- previous precision: 0.333
- previous recall: 0.333
- previous false positives: 2

to:

- new LOO F1: 0.400
- new precision: 0.500
- new recall: 0.333
- new false positives: 1

The new negatives improved calibration on false positives, but they did not add
the rescue positives needed to improve recall.

## Learning

The error-neighborhood selector is useful for hard-negative calibration, but the
current task pool is not yielding new rescue positives. The missed-positive
neighborhood around `BigCodeBench/368` and `BigCodeBench/963` produced universal
failures rather than DeepSeek or GLM rescues.

Before another fallback-head promotion attempt, prioritize acquisition sources
that have already shown at least one passing alternate, not merely similarity to
past fallback errors. A better next screen is to mine or fetch more
canonical-pass filesystem/archive tasks first, then run top-4 comparison only on
those plausibly solvable tasks.
