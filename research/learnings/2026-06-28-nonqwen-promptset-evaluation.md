# Non-Qwen Prompt-Set Evaluation

Run tag: `20260628-nonqwen-promptset-evaluation`

## What Changed

Added evaluator-backed scoring for the live non-Qwen prompt-set comparison.
The new evaluator replays each recorded live response against the materialized
SmokeCode/BigCodeBench task tests, extracts fenced Python when needed, and
writes one JSONL evaluation row per policy execution.

Artifacts:

- `tools/evaluate_orchestrated_prompt_set.py`
- `tests/test_evaluate_orchestrated_prompt_set.py`
- `research/evals/20260628-nonqwen-promptset-comparison-evaluation.jsonl`
- `research/evals/20260628-nonqwen-promptset-comparison-evaluation-report.json`

## Result

The trained task-level orchestrator solved 1 of 3 tasks. The fixed Qwen Coder
baseline solved 0 of 3 tasks on the same live responses.

The dependency-aware evaluation schema also reports pass rate over rows that
were actually evaluable in the current local environment.

| Policy | Passed | Pass Rate | Evaluable Passed | Evaluable Pass Rate | Missing Dependency Rows | Mean Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trained-orchestrator | 1 / 3 | 0.3333 | 1 / 1 | 1.0000 | 2 | 6955 ms |
| fixed-worker:ollama-cloud-qwen3-coder-480b | 0 / 3 | 0.0000 | 0 / 1 | 0.0000 | 2 | 4052 ms |

The useful positive signal is task `bigcodebench-hard-BigCodeBench-339`: the
orchestrator selected `ollama-cloud-glm-5.2`, and that response passed all 7
tests. The fixed Qwen response failed the same task tests.

## Dependency-Aware Caveat

Four policy execution rows are now explicitly marked as
`missing_eval_dependency` because the local evaluator environment did not have
imported benchmark dependencies available:

- `bigcodebench-hard-BigCodeBench-1004`: `matplotlib`
- `bigcodebench-hard-BigCodeBench-1053`: `pandas`

Those rows should be preserved as reproducibility failures for this harness,
but excluded from model-quality labels unless the dependency profile is pinned
or the task is rerun in an environment where the required packages exist.
Before promoting the router, rerun dependency-heavy tasks inside a pinned
benchmark environment or filter live comparison sets to tasks whose runtime
dependencies are available.

## Decision

Keep the evaluator and the result. This is the first small quality signal where
the trained router beats the fixed Qwen baseline on pass rate, and the
dependency-aware schema makes the signal easier to audit. It is still not a
promotion signal because the sample is tiny, slower, and partly coupled to
local dependency availability.

Turn-level routing remains deferred. The turn-level substrate code should stay
as roadmap-ready infrastructure until the task-level orchestrator has a larger
and more stable benchmark-backed dataset.
