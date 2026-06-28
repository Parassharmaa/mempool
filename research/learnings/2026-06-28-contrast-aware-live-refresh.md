# Contrast-Aware Live Refresh

Run tag: `20260628-contrast-aware-live-refresh`

## Question

Does the outcome-aware planner's next source, `contrast-aware`, produce stable
rows that can improve the active 23-task logits router?

## Live Run

Materialized and ran:

- `research/evals/20260628-contrast-aware-top4-repeat-manifest.json`
- `research/evals/20260628-contrast-aware-top4-repeat-tasks.json`

Run shape:

- tasks: `4`
- workers: GLM, DeepSeek, Kimi, Qwen
- repeat count: `2`
- total calls: `32`

Artifacts:

- `research/evals/results/20260628-contrast-aware-top4-repeat.jsonl`
- `research/evals/results/20260628-contrast-aware-top4-repeat-summary.json`
- `research/evals/results/20260628-contrast-aware-top4-repeat-audit.json`
- `research/datasets/20260628-contrast-aware-top4-repeat-routing.jsonl`
- `research/datasets/20260628-contrast-aware-top4-repeat-merge-ready-routing.jsonl`

## Outcome

The outcome audit passed with `32` rows, `4` tasks, `4` workers, and `2`
samples per worker-task.

Task results:

- `BigCodeBench-0`: broad-pass, Qwen target by latency
- `BigCodeBench-2`: Qwen target by latency, GLM also stable, Kimi unstable,
  DeepSeek failed
- `BigCodeBench-5`: GLM target by latency, all workers solved
- `BigCodeBench-8`: universal failure

The merge filter kept `3` stable rows and dropped the all-fail row.

## Refresh Candidate

Merged the `3` merge-ready rows into the active 23-task dataset:

- `research/datasets/20260628-contrast-aware-26task-routing.jsonl`

This adds:

- Qwen targets: `BigCodeBench-0`, `BigCodeBench-2`
- GLM target: `BigCodeBench-5`

The temperature selector initially exposed a stale API mismatch:
`leave_one_out_logits_evaluation` did not accept `l2` or return the
latency/solvable metrics expected by the refresh gate. This was repaired so the
training and LOO paths use the same regularization and metric surface.

Temperature sweep:

- `research/datasets/20260628-contrast-aware-26task-temperature-selection.json`

Best candidate remained quarantined. At reward temperature `0.05`:

- LOO target accuracy: `0.654`
- LOO solvable pass@1: `0.739`
- LOO mean latency regret: `800.3 ms`

The active baseline remains better:

- LOO target accuracy: `0.783`
- LOO solvable pass@1: `0.800`
- LOO mean latency regret: `501.1 ms`

## Evaluation

Focused tests:

- `PYTHONPATH=src python3 -m unittest tests.test_logits_router tests.test_select_logits_router_temperature`
- result: `11` tests passed

Full suite:

- `PYTHONPATH=src python3 -m unittest discover -s tests`
- result: `224` tests passed

Research-loop gate:

- `PYTHONPATH=src python3 tools/research_loop.py evaluate --tag 20260628-contrast-aware-live-refresh`
- result: `pass`
- score: `1.0`

## Decision

Keep the new data, LOO repair, and quarantine decision. Do not update
`research/policies/active_policy.json`.

The contrast-aware source is better than the prior non-Qwen-pressure source for
finding solvable rows, but the current linear logits router still cannot absorb
the new general/random latency targets without hurting held-out behavior. The
next modeling step should target latency-sensitive features or a richer
orchestrator head before another promotion attempt.
