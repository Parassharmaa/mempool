# Corpus-Guided Fallback Batch

Run tag: `20260628-corpus-guided-fallback-batch`

## Question

Can the mined fallback-opportunity corpus guide the next fresh benchmark
acquisition toward tasks likely to produce fallback positives or cheap
second-attempt rescues?

## Change

- Added `tools/select_corpus_guided_fallback_batch.py`.
- Added `tests/test_select_corpus_guided_fallback_batch.py`.
- Fixed the selector to accept repeated `--tasks` groups and to report the
  union of excluded task ids.
- Selected a fresh eight-task BigCodeBench-Hard acquisition batch from normal
  offset eligible pools while excluding all existing measured routing tasks and
  all mined fallback-corpus task ids.

## Source Signal

The selector uses the mined fallback corpus as an acquisition prior:

- corpus rows: 322
- unique corpus task ids: 86
- useful-any fallback seeds: 149
- useful-second fallback seeds: 78
- hard-negative seeds: 173

The first selection attempt against already-exhausted merged hard-profile pools
found zero fresh candidates. Rerunning against the normal-offset eligible pools
produced 65 fresh candidates.

## Selected Batch

Output:

- `research/evals/bigcodebench_hard_corpus_guided_fallback_batch8_tasks.json`
- `research/evals/bigcodebench_hard_corpus_guided_fallback_batch8_report.json`

Selected task ids:

- `bigcodebench-hard-BigCodeBench-327`
- `bigcodebench-hard-BigCodeBench-281`
- `bigcodebench-hard-BigCodeBench-322`
- `bigcodebench-hard-BigCodeBench-339`
- `bigcodebench-hard-BigCodeBench-671`
- `bigcodebench-hard-BigCodeBench-675`
- `bigcodebench-hard-BigCodeBench-539`
- `bigcodebench-hard-BigCodeBench-673`

The selected rows are mostly filesystem/text/general tasks with low active-router
first-second margins. Several deliberately sit on Kimi/Qwen boundaries, which is
useful for testing whether fallback behavior can rescue first-attempt mistakes
without simply adding more duplicate Qwen-first evidence.

## Decision

Keep:

- corpus-guided acquisition selector
- selected eight-task acquisition batch
- report-level ranking data for later postmortem comparison

Do not promote:

- any new router or fallback policy from this run alone

This run only chooses the next labels to buy; it does not evaluate worker
outcomes.

## Next

Evaluate the selected batch across the top worker pool with repeated samples,
convert the resulting outcomes into routing records, then check whether the
batch adds fallback-positive examples or cheap second-attempt rescues before
training another value-aware fallback candidate.
