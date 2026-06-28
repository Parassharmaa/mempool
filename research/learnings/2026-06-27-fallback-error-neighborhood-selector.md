# Fallback Error Neighborhood Selector

Run tag: `20260627-fallback-error-neighborhood-selector`

## What Changed

Added `tools/select_fallback_error_neighborhood_batch.py`, a targeted batch
selector for the mined fallback-action head's leave-one-out errors.

The selector reads:

- mined fallback cases
- mined-head leave-one-out report
- available BigCodeBench task pools
- active router registry

It extracts two error neighborhoods:

- false negatives: useful fallbacks the head missed
- false positives: hard negatives the head incorrectly wanted to retry

Then it selects fresh tasks near both neighborhoods, excluding all tasks already
present in routing datasets.

## Result

The generated batch is:

- tasks: `research/evals/bigcodebench_hard_fallback_error_neighborhood_batch8_tasks.json`
- report: `research/evals/bigcodebench_hard_fallback_error_neighborhood_batch8_report.json`

The selector considered 30 fresh candidate tasks after excluding 49 previously
routed or smoke tasks, then emitted 8 tasks:

- `BigCodeBench/988`: missed-positive neighborhood
- `BigCodeBench/765`: missed-positive neighborhood
- `BigCodeBench/800`: missed-positive neighborhood
- `BigCodeBench/985`: missed-positive neighborhood
- `BigCodeBench/492`: false-positive neighborhood
- `BigCodeBench/579`: false-positive neighborhood
- `BigCodeBench/120`: false-positive neighborhood
- `BigCodeBench/528`: false-positive neighborhood

The missed-positive side targets filesystem/subprocess/filesystem-data tasks
near the missed rescue positives `BigCodeBench/368` and `BigCodeBench/963`.
The false-positive side targets tasks near `BigCodeBench/857` and
`BigCodeBench/594`, where the head over-predicted fallback usefulness.

## Learning

The fallback head's next data should not be a generic uncertainty batch. It
should deliberately sample near the observed error surface:

- likely rescue cases around missed positives
- hard negatives around false positives

This gives the next cloud run a clearer purpose: either recover more DeepSeek
or GLM rescue labels, or prove that the current fallback-action features are
not enough to separate these neighborhoods.

## Next Step

Run a small top-4 worker screen on this batch, preferably starting with 4 tasks:
two from the missed-positive side and two from the false-positive side. Convert
the outcomes back into mined fallback cases, retrain the mined head, and compare
leave-one-out F1 against the current `0.333` baseline.
