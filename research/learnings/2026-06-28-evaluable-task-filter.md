# Current-Environment Evaluable Task Filter

Run tag: `20260628-evaluable-task-filter`

## What Changed

Added a preflight filter for materialized SmokeCode/BigCodeBench task files.
The filter extracts root imports from task prompts and tests, checks whether
those modules are importable in the current Python environment, and writes:

- a filtered task file containing only locally evaluable tasks
- a report explaining which tasks were excluded and why

Artifacts:

- `tools/filter_evaluable_tasks.py`
- `tests/test_filter_evaluable_tasks.py`
- `research/evals/20260628-current-env-evaluable-tasks.json`
- `research/evals/20260628-current-env-evaluable-tasks-report.json`

## Result

The union of the two task files used by the live non-Qwen comparison contains
40 unique tasks. The preflight found 27 tasks evaluable in the current local
environment and excluded 13 tasks with missing imports.

Missing import roots:

- `bs4`
- `matplotlib`
- `numpy`
- `pandas`
- `requests`
- `sklearn`

This matches the previous evaluator-backed finding: task
`bigcodebench-hard-BigCodeBench-339` remains locally evaluable, while
`bigcodebench-hard-BigCodeBench-1004` and
`bigcodebench-hard-BigCodeBench-1053` are excluded by the preflight.

## Decision

Keep the filter and the manifest. Future live prompt-set comparisons should use
the filtered task pool, or a pinned benchmark environment, before turning rows
into model-quality routing labels.

This does not replace dependency-heavy benchmark work. It gives the current
task-level orchestrator loop a clean acquisition lane while the broader
benchmark dependency profile remains a separate infrastructure task.
