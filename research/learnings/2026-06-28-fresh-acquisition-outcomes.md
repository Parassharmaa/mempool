# Fresh Acquisition Outcomes

Run tag: `20260628-fresh-acquisition-outcomes`

## Question

Do the six value-head-selected normal BigCodeBench acquisition tasks produce usable real-worker training signal, and does adding the merge-ready rows improve the small multi-head orchestrator?

## Result

Keep the outcome data, quarantine the trained 54-task policy candidate.

The bounded real-worker run evaluated six selected tasks across the current top-four cloud worker pool:

- `ollama-cloud-glm-5.2`
- `ollama-cloud-deepseek-v4-pro`
- `ollama-cloud-kimi-k2.7-code`
- `ollama-cloud-qwen3-coder-480b`

It produced `24` outcome rows, `6` tasks, and `4` workers per task. The outcome audit was conversion-ready.

## Outcome Signal

- GLM solved `4/6`.
- DeepSeek solved `4/6`.
- Kimi solved `3/6`.
- Qwen solved `4/6`.
- Tasks `BigCodeBench-12` and `BigCodeBench-6` were all-fail across the top-four workers.
- Tasks `BigCodeBench-1`, `BigCodeBench-14`, and `BigCodeBench-4` were all-pass.
- Task `BigCodeBench-7` was the useful specialist contrast: GLM, DeepSeek, and Qwen passed, while Kimi failed.

Filtering for merge-ready routing rows kept `4/6` tasks and dropped the two all-fail rows.

## Candidate Router

The four merge-ready records were merged with the existing 50-task experimental routing set and exported as a 54-record small-orchestrator substrate.

Target mix:

- DeepSeek: `11`
- GLM: `3`
- Kimi: `10`
- Qwen: `30`

The trained candidate looked acceptable in-sample, but the leave-one-out gate rejected it:

- Candidate LOO target accuracy: `0.537`
- Baseline 50-task LOO target accuracy: `0.600`
- Candidate LOO pass@1: `0.778`
- Candidate LOO solvable pass@1: `0.824`
- Candidate LOO mean latency regret: `1723.1 ms`
- Gate decision: `quarantine`
- Gate reason: target accuracy below the `0.550` minimum

## Interpretation

This batch improves the outcome ledger but does not justify a policy refresh. The new rows mostly add easy all-pass examples plus two all-fail diagnostics; only one task is a strong worker-differentiating contrast. That explains why the broader candidate preserved solvable pass@1 above `0.80` but weakened target-worker accuracy.

The selector should be made more contrast-aware before the next acquisition pass. In particular, it should down-rank likely all-pass tasks and prioritize tasks where prior features imply worker disagreement, not just value-head uncertainty.

## Artifacts

- Selected task batch: `research/evals/20260628-fresh-pool-frontier-value-head-tasks.json`
- Real-worker summary: `research/evals/results/20260628-fresh-acquisition-outcomes.json`
- Real-worker outcomes: `research/evals/results/20260628-fresh-acquisition-outcomes.jsonl`
- Outcome audit: `research/evals/results/20260628-fresh-acquisition-outcomes_audit.json`
- Raw routing dataset: `research/datasets/20260628-fresh-acquisition-outcomes-routing.jsonl`
- Merge-ready routing dataset: `research/datasets/20260628-fresh-acquisition-outcomes-routing-merge-ready.jsonl`
- Merge-ready report: `research/datasets/20260628-fresh-acquisition-outcomes-routing-merge-ready-report.json`
- 54-task routing candidate: `research/datasets/20260628-fresh-acquisition-54task-routing.jsonl`
- 54-task substrate: `research/datasets/20260628-fresh-acquisition-54task-substrate.jsonl`
- 54-task model: `research/models/20260628-fresh-acquisition-54task-multihead.json`
- 54-task report: `research/evals/results/20260628-fresh-acquisition-54task-multihead-report.json`
- Policy gate: `research/evals/results/20260628-fresh-acquisition-54task-policy-gate.json`

## Next Step

Add a contrast-aware acquisition score that penalizes likely all-pass/all-fail tasks and favors expected disagreement among the top-four workers, then rerun selection before spending more real-worker calls.
