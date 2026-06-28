# Direct Specialist Mining

Run tag: `20260628-direct-specialist-mining`

Goal: stop spending full comparison calls on fresh similarity candidates unless
a specialist first proves it can solve the task at least once.

Change:

Added `tools/plan_direct_specialist_mining.py`.

The planner:

- selects fresh tasks from a low-risk pool
- excludes existing routing records, outcome files, and screening summaries
- runs a single specialist worker pool with one sample per task
- emits a run manifest plus summary command
- leaves graduation to `tools/select_passed_tasks_from_outcomes.py`

Screen run:

Ran DeepSeek V4 Pro on two fresh low-risk candidates:

- `BigCodeBench/124`
- `BigCodeBench/324`

Result:

- DeepSeek: `0/2`
- Failure modes: `2` `test_failure`
- Graduated tasks: `0`
- Mean latency: `9536 ms`

Decision: keep the direct mining planner.

The data result is negative, but the workflow is now the right cheap gate. It
cost two calls to learn that these candidates should not enter the repeated
comparison pool. This is exactly the acquisition shape we want before spending
top-4 comparison calls.

Next step:

Continue direct mining in tiny batches, but bias toward task pools that already
produced positives in prior mining (`qwen_positive`, `kimi_positive`,
`specialist_positive`) rather than only "fresh low-risk" candidates from the
merged eligible pool.

Prior-pool follow-up:

The previously positive Qwen-negative-untrained pool had no remaining
unscreened DeepSeek candidates after excluding prior outcome files and today's
low-risk screen. The remaining DeepSeek-like pool produced one unscreened
candidate:

- `BigCodeBench/800`

Result:

- DeepSeek: `0/1`
- Failure mode: `test_failure`
- Graduated tasks: `0`
- Mean latency: `4351 ms`

The failed sample treated dictionary inputs as numeric scalars, so this was a
real behavioral miss rather than a harness parse issue. This closes the current
direct DeepSeek prior-pool screen with no new positives.

Updated decision:

Keep the planner, but do not keep mining the same exhausted pools. The next
useful acquisition step should either widen to a new task source or use a
different cheap filter, such as small multi-worker screen on tasks with stronger
solvability priors, before spending repeated comparison calls.

Artifacts:

- `tools/plan_direct_specialist_mining.py`
- `research/programs/20260628-direct-deepseek-mining-lowrisk1.json`
- `research/evals/20260628-direct-deepseek-mining-lowrisk1-tasks.json`
- `research/evals/results/20260628-direct-deepseek-mining-lowrisk1-summary.json`
- `research/evals/20260628-direct-deepseek-mining-lowrisk1-graduated-tasks.json`
- `research/programs/20260628-direct-deepseek-priorpool2.json`
- `research/programs/20260628-direct-deepseek-like-remaining1.json`
- `research/evals/20260628-direct-deepseek-like-remaining1-tasks.json`
- `research/evals/results/20260628-direct-deepseek-like-remaining1-summary.json`
- `research/evals/20260628-direct-deepseek-like-remaining1-graduated-tasks.json`
