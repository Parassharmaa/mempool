# Canonical Specialist Selector

Run tag: `20260627-canonical-specialist-selector`

## What Changed

Added `tools/select_canonical_specialist_batch.py`, a selector for fresh
BigCodeBench tasks that come from canonical-pass eligible pools and place a
specialist worker near the active router's top route.

The selector scores candidates by:

- filesystem/archive signal
- low environment risk
- GLM or DeepSeek rank/probability under the active router
- low top-vs-second margin
- prior routing/outcome exclusion

This is intended to avoid repeating the previous error-neighborhood screen,
which selected similar tasks but produced universal failures.

## Result

The selector read the current eligible task pools and excluded all tasks already
present in routing datasets.

It found:

- 26 fresh candidates
- 53 excluded previously seen tasks
- 8 selected canonical-specialist candidates

Selected batch:

- `BigCodeBench/800`: filesystem, DeepSeek second, low margin
- `BigCodeBench/528`: filesystem/plotting/datasci, DeepSeek top
- `BigCodeBench/985`: filesystem/datasci, specialist rank 3
- `BigCodeBench/162`: plotting/datasci, DeepSeek second
- `BigCodeBench/399`: plotting/datasci, DeepSeek second
- `BigCodeBench/897`: plotting/datasci, DeepSeek second
- `BigCodeBench/161`: network/filesystem/datasci, specialist rank 3
- `BigCodeBench/1012`: network/filesystem/archive, specialist rank 3

Artifacts:

- batch tasks: `research/evals/bigcodebench_hard_canonical_specialist_batch8_tasks.json`
- batch report: `research/evals/bigcodebench_hard_canonical_specialist_batch8_report.json`
- next screen tasks: `research/evals/bigcodebench_hard_canonical_specialist_screen4_tasks.json`
- next screen report: `research/evals/bigcodebench_hard_canonical_specialist_screen4_report.json`

## Learning

The available fresh pool still has canonical-pass tasks with meaningful
specialist pressure, especially around DeepSeek. The next top-4 screen should
start with `800`, `528`, `985`, and `162` because they combine eligible-source
provenance with either filesystem signal or a close DeepSeek route.

This does not guarantee new rescue positives, but it is a better acquisition
criterion than pure fallback-error similarity because the source pools have
already passed canonical-solution filtering.
