# Mined Fallback Head

Run tag: `20260627-mined-fallback-head`

## What Changed

Added a mined fallback-action head that trains on historical fallback cases
instead of full routing records. The head predicts whether the orchestrator
should keep searching after the active router's top worker fails.

Artifacts:

- model: `research/models/20260627-mined-fallback-head.json`
- report: `research/evals/results/20260627-mined-fallback-head-report.json`
- source dataset: `research/datasets/20260627-historical-fallback-cases.jsonl`

The training tool deduplicates by task id by default, so repeated hard-negative
rows from multiple source datasets do not count as extra independent evidence.

## Result

The deduplicated training set has:

- 16 unique fallback-opportunity tasks
- 3 useful fallback positives
- 13 hard negatives

Training fit the tiny dataset perfectly:

- training F1: 1.00
- training precision: 1.00
- training recall: 1.00

Leave-one-out evaluation was much weaker:

- LOO F1: 0.333
- LOO precision: 0.333
- LOO recall: 0.333
- LOO accuracy: 0.75
- true positives: 1
- false positives: 2
- false negatives: 2

The head only generalized to `BigCodeBench/526`. It missed `BigCodeBench/368`
and `BigCodeBench/963`, and it falsely predicted fallbacks for
`BigCodeBench/857` and `BigCodeBench/594`.

## Learning

The mined fallback dataset is useful because it gives the action head real
positive labels, but it is still too small and too heterogeneous for a
promotion-ready policy. The train-vs-LOO gap is a clear memorization signal.

The next useful data acquisition should target neighborhoods around the missed
positive rescues and false positives:

- more DeepSeek rescue cases near `BigCodeBench/368` and `BigCodeBench/526`
- more GLM rescue cases near `BigCodeBench/963`
- more hard negatives near `BigCodeBench/857` and `BigCodeBench/594`

Do not promote this mined fallback head into the active runtime yet. Use it as a
calibration artifact and as a guide for the next fallback-focused batch.
