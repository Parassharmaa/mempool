# Active Rescue Top-4 Repeat

Run tag: `20260628-active-rescue-top4-repeat`

This run evaluated an eight-task active-rescue acquisition batch against the
current top-four cloud worker pool: `Qwen`, `DeepSeek`, `Kimi`, and `GLM`.

## Results

- Samples: 64
- Tasks: 8
- Solvable tasks: 4 (`BigCodeBench/283`, `BigCodeBench/370`, `BigCodeBench/644`, `BigCodeBench/672`)
- Universal failures: 4 (`BigCodeBench/125`, `BigCodeBench/365`, `BigCodeBench/397`, `BigCodeBench/565`)
- Worker pass rates:
  - `Qwen`: 8 / 16
  - `DeepSeek`: 6 / 16
  - `Kimi`: 6 / 16
  - `GLM`: 5 / 16

The fallback miner found four fallback opportunities, but every one was a hard
negative: no useful any-fallback rows and no useful second-fallback rows.

## Decision

Keep the run as negative evidence and as task-level routing signal. It improves
the training substrate, but it does not justify promoting a new active fallback
policy.

The current publishable training artifact is the 66-task multi-head
orchestrator checkpoint:

- Substrate: `research/datasets/20260628-m5-current-task-66task-substrate.jsonl`
- Model: `research/models/20260628-m5-current-task-66task-multihead.json`
- Report: `research/evals/20260628-m5-current-task-66task-multihead-report.json`
