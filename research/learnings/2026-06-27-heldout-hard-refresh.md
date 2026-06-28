# Held-Out Hard Refresh

## Question

Can the held-out hard-batch evidence be folded into the active logits router
without letting all-fail fastest-failure rows hide a miss on solvable tasks?

## Setup

Merged the active 19-task routing dataset with the four held-out hard records:

- input: `research/datasets/20260627-mixed-winner-19task-routing.jsonl`
- input: `research/datasets/20260627-heldout-hard-active-policy-routing.jsonl`
- output: `research/datasets/20260627-mixed-winner-23task-heldout-hard-routing.jsonl`

Ran reward-temperature selection with the new solvable-row gate:

- temperatures: 0.05, 0.10, 0.20, 0.50
- minimum LOO target accuracy: 0.75
- maximum LOO target-accuracy drop: 0.10
- minimum LOO solvable pass@1: 0.75
- maximum LOO latency regret: 1000 ms
- maximum LOO latency-regret increase: 500 ms

## Result

Temperature 0.05 passed the gate and was promoted:

- model:
  `research/models/20260627-mixed-winner-23task-heldout-hard-reward-t0p05-logits-router.json`
- dataset:
  `research/datasets/20260627-mixed-winner-23task-heldout-hard-routing.jsonl`
- LOO target accuracy: 0.7826
- LOO pass@1: 0.6957
- LOO solvable pass@1: 0.8000
- LOO solvable target accuracy: 0.7500
- LOO mean latency regret: 501.1 ms

The refresh has one warning: target accuracy dropped by 0.007 versus the active
19-task baseline, which is inside the allowed gate.

Direct active-policy evaluation after promotion:

- merged 23-task dataset: 18/23 solved, pass@1 0.7826, solvable pass@1 0.9000
- held-out hard slice: 1/4 solved, target accuracy 1.0, solvable pass@1 1.0

The held-out hard slice is no longer held out after promotion; treat it as a
regression slice from here onward.

## Interpretation

The solvable-row gate did its job. The previous active router missed the only
solvable held-out row by routing everything to Qwen. The promoted 23-task router
routes that row to DeepSeek while preserving target-worker diversity across
Qwen, Kimi, GLM, and DeepSeek.

This is still a linear logits router over engineered task features. It is a
stronger M3 policy artifact, not the final small neural or language-model
orchestrator.

## Next Step

Run one more fresh held-out batch after this promotion. If the same Qwen-default
failure recurs, the next improvement should be model capacity or a second-stage
verifier/abstain head rather than more single-row feature patches.
