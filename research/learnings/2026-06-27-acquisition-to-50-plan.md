# Acquisition To 50 Plan

The M5 readiness gate now blocks only on data volume: active rows are 23 and
the threshold is 50. Added a planner for the next acquisition wave:

- `src/mempool/acquisition_to_50.py`
- `tools/plan_acquisition_to_50.py`
- `research/programs/acquisition_to_50_plan.json`
- `research/evals/bigcodebench_hard_acquisition_to_50_wave1_tasks.json`

The generated wave selects 41 fresh BigCodeBench tasks from existing top-four
specialist, hard, fresh, offset, and eligible pools. This overselects against
the 27-row gap because unstable all-fail or partial-pass rows must be filtered
out before merging.

Wave 1 uses the measured top-four pool:

- `ollama-cloud-qwen3-coder-480b`
- `ollama-cloud-kimi-k2.7-code`
- `ollama-cloud-glm-5.2`
- `ollama-cloud-deepseek-v4-pro`

Expected calls: 41 tasks x 4 workers x 2 repeats = 328 model calls.

Guardrails:

- run with `.venv-bigcodebench`
- require `numpy` and `pandas`
- audit outcome rows before conversion
- build repeated routing rows with evaluator-package filters
- run merge-readiness audit
- use guarded merge with `--require-merge-ready`

Next step: execute the `run_repeated_eval` command from
`research/programs/acquisition_to_50_plan.json`, then run the remaining
commands in order.
