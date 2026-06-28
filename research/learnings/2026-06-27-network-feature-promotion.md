# Network Feature Promotion

## Question

Can explicit network/archive/request interaction features make the 19-task
Qwen/Kimi/GLM/DeepSeek routing dataset promotable?

## Result

Yes. Adding compact interaction features to `src/mempool/task_features.py`
converted the 19-task hard-selector dataset from a near miss into a promoted
policy.

New feature examples:

- `signal_network`
- `signal_archive`
- `signal_plotting`
- `combo_network_archive`
- `combo_network_plotting`
- `combo_network_filesystem`

The selected model is:

- `research/models/20260627-mixed-winner-19task-network-features-reward-t0p05-logits-router.json`

The active dataset is:

- `research/datasets/20260627-mixed-winner-19task-routing.jsonl`

## Gate Result

The selected reward-temperature candidate uses temperature 0.05 and passed the
refresh gate with warnings:

- LOO target accuracy: 0.7895
- LOO pass@1: 0.8421
- LOO mean latency regret: 606.6 ms
- Target mix: Qwen 13, Kimi 4, GLM 1, DeepSeek 1

The prior 10-task active policy had 0.8 LOO target accuracy, 0.9 pass@1, and
518.6 ms latency regret. The new policy adds 9 tasks and one more target worker
while staying inside the allowed regression gates.

## Operational Comparison

On the 19-task dataset:

- Active logits router: 17/19 solved, 0.8421 target accuracy.
- Fastest single worker: 13/19 solved, 0.6842 target accuracy.
- Strongest single worker: 14/19 solved, 0.0526 target accuracy.
- Oracle target: 19/19 solved.

## Decision

Promote the 19-task network-feature logits router in
`research/policies/active_policy.json`. The next step should be a held-out hard
top-four diagnostic, not a larger backbone yet.
