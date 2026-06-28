# Held-Out Hard Active-Policy Diagnostic

## Question

Does the promoted 19-task network-feature logits router generalize to fresh
top-four BigCodeBench-Hard tasks that were excluded from all prior routing
datasets and outcome files?

## Setup

Selected four fresh hard-strategy tasks from
`research/evals/bigcodebench_hard_top4_eligible_merged_tasks.json`, excluding
all existing `research/datasets/*routing.jsonl` rows and prior outcome JSONL
task IDs:

- `BigCodeBench/100`
- `BigCodeBench/763`
- `BigCodeBench/1022`
- `BigCodeBench/955`

Each task was run twice per worker across GLM 5.2, DeepSeek V4 Pro, Kimi K2.7
Code, and Qwen3 Coder 480B.

## Result

Only `BigCodeBench/763` was solved:

- DeepSeek V4 Pro: 2/2, mean latency 6076.5 ms
- Kimi K2.7 Code: 1/2, mean latency 6522.0 ms
- GLM 5.2: 0/2
- Qwen3 Coder 480B: 0/2

The active logits router predicted Qwen for all four held-out rows. This gave:

- pass@1: 0.0
- solvable-row pass@1: 0.0 over 1 solvable row
- target accuracy: 0.75
- mean latency regret: 0.0 ms

The target-accuracy number is misleading because the three unsolved rows target
the fastest failure path, which is Qwen. The only row where a correct solution
exists is a DeepSeek target, and the active router misses it.

## Interpretation

This is a useful negative result. The current features learned the cheap/fast
Qwen default too strongly for data/filesystem tasks. Target accuracy and latency
regret are insufficient held-out gates when many rows are all-fail; pass@1 on
solvable rows must be reported separately before promotion.

The evaluation code now reports `solvable_task_count`, `solvable_pass_at_1`,
and `solvable_target_accuracy` for both active logits-router reports and
baseline comparison reports. The refresh gate and temperature selector also
accept an optional `min_loo_solvable_pass_at_1` threshold.

## Next Step

Merge this held-out DeepSeek target into a candidate training dataset only as a
quarantined refresh, using the new solvable-row gate before any promotion.
