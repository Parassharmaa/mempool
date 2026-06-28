# Active Rescue Acquisition Selector

Run tag: `20260628-active-rescue-acquisition-selector`

## Question

Can we select fresh benchmark tasks that are more likely to produce
active-router top-fail and alternate-pass rescue labels than generic low-margin
fallback acquisition?

## Change

- Added `tools/select_active_rescue_acquisition_batch.py`.
- Added `tests/test_select_active_rescue_acquisition_batch.py`.
- Selected a fresh eight-task BigCodeBench-Hard batch from the remaining normal
  offset eligible pools.

The selector scores candidates using conditional fallback evidence from the
mined corpus:

- same active top worker as historical useful fallbacks
- same top-worker to alternate-worker rescue pair
- same top-worker to second-worker useful-second pair
- similarity to useful fallback prompts
- penalty for same-pair and same-top hard negatives
- active-router low-margin uncertainty

## Selected Batch

Output:

- `research/evals/20260628-active-rescue-acquisition-batch8-tasks.json`
- `research/evals/20260628-active-rescue-acquisition-batch8-report.json`

Selected task ids:

- `bigcodebench-hard-BigCodeBench-283`
- `bigcodebench-hard-BigCodeBench-365`
- `bigcodebench-hard-BigCodeBench-125`
- `bigcodebench-hard-BigCodeBench-672`
- `bigcodebench-hard-BigCodeBench-565`
- `bigcodebench-hard-BigCodeBench-644`
- `bigcodebench-hard-BigCodeBench-397`
- `bigcodebench-hard-BigCodeBench-370`

The strongest selected row is `BigCodeBench/283`: active top worker Qwen,
second worker DeepSeek, low margin 0.081, positive Qwen-to-DeepSeek rescue
similarity 5.9, and hard same-pair similarity only 0.1.

The remaining batch includes Kimi-to-Qwen and Qwen-to-Kimi boundary tasks. Some
of these still have meaningful hard-negative similarity, so they are evaluation
targets rather than assumed wins.

## Decision

Keep:

- active-rescue acquisition selector
- selected eight-task acquisition batch
- pair-level rescue and hard-negative scoring details in the report

Do not promote:

- any policy from this selection-only run

## Next

Run the selected batch with repeated top-four worker calls. The key success
criterion is not raw pass rate; it is whether mining the resulting routing
dataset yields useful active-router fallback positives, especially second-rank
rescues.
