# 2026-06-27 Offset-99 Filesystem Boundary

## Question

Can we add more filesystem-heavy BigCodeBench tasks to separate Qwen-only,
Kimi-favored, and other worker-winning regions?

## Setup

Eligible scan:

- Start offset: 99
- Output: `research/evals/bigcodebench_hard_eligible_offset99_tasks.json`
- Report: `research/evals/bigcodebench_hard_eligible_offset99_report.json`

Mining:

- Qwen outcomes: `research/evals/results/20260627-qwen-offset99-mining-8.jsonl`
- Kimi outcomes: `research/evals/results/20260627-kimi-offset99-mining-8.jsonl`
- Qwen-only candidates:
  `research/evals/bigcodebench_hard_offset99_qwen_only_tasks.json`

Repeated comparison:

- Outcomes:
  `research/evals/results/20260627-qwen-only-offset99-repeat-compare.jsonl`
- Summary:
  `research/evals/results/20260627-qwen-only-offset99-repeat-compare-summary.json`
- Routing dataset:
  `research/datasets/20260627-qwen-only-offset99-repeat-compare-routing.jsonl`

Expanded model:

- Dataset: `research/datasets/20260627-mixed-winner-8task-routing.jsonl`
- Model: `research/models/20260627-mixed-winner-8task-logits-router.json`
- Report: `research/datasets/20260627-mixed-winner-8task-logits-router-report.json`

## Result

The offset-99 scan found 8 new canonical-pass tasks, mostly filesystem and
subprocess:

- 785, 800, 854, 857, 906, 928, 963, 988

Single-sample mining found:

- Qwen: 4/8, passing 854, 906, 928, 963
- Kimi: 2/8, passing 906, 928

The apparent Qwen-only tasks were 854 and 963. Repeated comparison changed the
picture:

- 854: all four workers were 2/2, so Qwen wins mostly by latency.
- 963: GLM was 1/2 while DeepSeek, Kimi, and Qwen were 0/2, making GLM the
  empirical hard target.

The merged eight-task dataset now has three target workers:

- Qwen Coder: 5 tasks
- Kimi K2.7 Code: 2 tasks
- GLM 5.2: 1 task

The eight-task logits router fits the training set:

- Training target accuracy: 8/8
- Training pass@1: 8/8
- Evaluation mean KL: 0.0015660431860364948

Leave-one-out is less stable:

- LOO target accuracy: 6/8
- LOO pass@1: 6/8
- Misses: 454 and 963, both routed to Kimi when held out

## Decision

Adding a single GLM-target task is useful but not enough. The router can fit a
three-worker target set, but held-out evaluation shows that sparse GLM and
Qwen-only filesystem regions collapse toward Kimi. Continue mining filesystem
tasks, especially GLM-positive/Qwen-negative and Qwen-only/Kimi-negative
examples, before increasing model complexity.
