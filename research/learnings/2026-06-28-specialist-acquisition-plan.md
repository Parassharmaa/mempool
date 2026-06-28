# Specialist Acquisition Plan

Run tag: `20260628-specialist-acquisition-plan`

Goal: turn the specialist miss audit into a measured data-acquisition queue
instead of adding more hand-written routing features.

Finding:

The current latency-aware multi-head candidate still misses specialist targets
by routing them back to the broad Qwen region. The useful next signal is not a
new feature patch; it is fresh measured rows near the current DeepSeek, GLM, and
Kimi miss clusters.

Change:

Added a specialist acquisition planner that:

- normalizes BigCodeBench task ids across source files and routing datasets
- reads leave-one-out misses from a candidate report
- groups misses by target specialist worker
- ranks fresh BigCodeBench tasks by library/category/risk similarity to each
  worker's miss seeds
- excludes the existing 50-task routing dataset
- emits both a task file and a manifest for the next benchmark run

The planner also accepts comparison workers. For this run, the selected
specialists are DeepSeek V4 Pro, GLM 5.2, and Kimi K2.7 Code, with Qwen3 Coder
480B included as the incumbent comparison worker. That matters because the new
rows should prove whether a specialist actually beats the current default, not
just whether it can solve a nearby task in isolation.

Result:

- Selected tasks: `12`
- Workers to run: `4`
- Repeats per worker/task: `2`
- Planned calls: `96`
- DeepSeek-focused tasks: `208`, `942`, `636`, `513`
- GLM-focused tasks: `528`, `985`, `988`, `857`
- Kimi-focused tasks: `211`, `1129`, `1012`, `1039`

Decision: keep planner and manifest.

Next step:

Run the planned task file against the specialist worker pool plus Qwen. Convert
the repeated outcomes into routing records only after checking empirical
pass-rate and latency stability. If the new rows produce confirmed non-Qwen
targets, merge them into the orchestrator substrate and re-run the policy gate.

First bounded smoke:

Ran only the first planned task, `BigCodeBench/208`, with two repeats across the
four planned workers. All eight attempts failed with `test_failure`:

- GLM 5.2: `0/2`, mean latency `5840.5 ms`
- DeepSeek V4 Pro: `0/2`, mean latency `8945.5 ms`
- Kimi K2.7 Code: `0/2`, mean latency `12810.5 ms`
- Qwen3 Coder 480B: `0/2`, mean latency `18956.0 ms`

Decision: do not merge this bounded smoke row into the routing dataset. The
first selected DeepSeek-neighborhood task appears to be a universal failure
under the current worker/evaluator setup, so the full acquisition run should
either skip universal-failure candidates after a quick screen or continue with
the remaining tasks and filter out all-zero rows before training.

Artifacts:

- `tools/plan_specialist_acquisition.py`
- `src/mempool/acquisition_plan.py`
- `research/programs/20260628-specialist-acquisition-plan.json`
- `research/evals/20260628-specialist-acquisition-tasks.json`
- `research/evals/results/20260628-specialist-acquisition-limit1.json`
- `research/evals/results/20260628-specialist-acquisition-limit1.jsonl`
