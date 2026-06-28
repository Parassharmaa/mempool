# Qwen-Fail Contrast Selector

Run tag: `20260628-qwen-fail-contrast-selector`

## Question

Can a contrastive acquisition heuristic find tasks that look closer to known
Qwen-fail specialist rows than to nearby Qwen-fast anchor rows?

## Change

Added a dedicated selector:

- `tools/select_qwen_fail_contrast_batch.py`
- `tests/test_select_qwen_fail_contrast_batch.py`

The selector scores fresh tasks by:

- similarity to Qwen-fail specialist seeds
- negative similarity to Qwen-fast anchor seeds
- active-router uncertainty
- active-router probability mass on specialist workers
- task risk and plausibility penalties

For this run:

- Qwen-fail seed: `BigCodeBench/917`
- Qwen-fast anchors: `BigCodeBench/92`, `184`, `1003`, `34`
- task source:
  `research/evals/bigcodebench_hard_next3_profile_eligible_merged_tasks.json`

## Planning Result

After excluding prior routing and outcome ledgers, only one fresh next3
candidate remained:

- `BigCodeBench/1008`

The contrast score was negative because `1008` was still more similar to the
Qwen-fast anchor set than to `917`, but it was screened because it closed the
remaining next3 frontier.

Artifacts:

- plan:
  `research/evals/20260628-qwen-fail-contrast-selector-report.json`
- tasks:
  `research/evals/20260628-qwen-fail-contrast-selector-tasks.json`

## Screen Result

One-sample top-four screen:

- outcomes:
  `research/evals/results/20260628-qwen-fail-contrast-selector-screen1.jsonl`
- summary:
  `research/evals/results/20260628-qwen-fail-contrast-selector-screen1-summary.json`

`BigCodeBench/1008` was a useful Qwen-fail candidate:

- DeepSeek passed
- Kimi passed
- Qwen failed
- GLM failed

## Repeat Result

`1008` was repeated across top-four with two samples per worker:

- outcomes:
  `research/evals/results/20260628-qwen-fail-contrast-selector-positive-repeat.jsonl`
- summary:
  `research/evals/results/20260628-qwen-fail-contrast-selector-positive-repeat-summary.json`
- routing row:
  `research/datasets/20260628-qwen-fail-contrast-selector-positive-repeat-routing.jsonl`

Stable result:

- DeepSeek: 2/2, mean latency 6581.5 ms, target worker
- Kimi: 2/2, mean latency 12556.5 ms
- Qwen: 0/2
- GLM: 0/2

The active 23-task router predicted Qwen on `1008`, so it missed pass@1:

- `research/evals/results/20260628-qwen-fail-contrast-active-policy-on-1008.json`

## Refresh Attempt

The run tested a 25-task candidate dataset:

- active 23-task dataset
- `917` Kimi target
- `1008` DeepSeek target

Artifacts:

- dataset:
  `research/datasets/20260628-mixed-winner-25task-qwen-fail-contrast-routing.jsonl`
- temperature selection:
  `research/datasets/20260628-mixed-winner-25task-qwen-fail-contrast-temperature-selection.json`
- refresh decision:
  `research/policies/20260628-mixed-winner-25task-qwen-fail-contrast-refresh.json`

Decision: quarantine. The best candidate was temperature `0.05`, but it had:

- LOO target accuracy: 0.560
- LOO solvable pass@1: 0.636
- LOO mean latency regret: 4655.4 ms

This is worse than the active 23-task policy and fails the promotion gate.

## Decision

Keep the selector, the `1008` routing row, and the learning. Do not promote the
25-task router.

`1008` is valuable because it adds a second stable Qwen-fail specialist target,
this time for DeepSeek rather than Kimi. The failed refresh suggests the current
linear logits router and prompt-feature set are too coarse to absorb these
exceptions safely. The next improvement should target feature expressiveness or
a richer small head, not another temperature-only refresh.
