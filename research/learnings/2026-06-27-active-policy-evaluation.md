# 2026-06-27 Active Policy Evaluation

## Question

Can evaluation load the promoted active policy from the registry instead of
hardcoding a logits-router model path?

## Setup

- Registry: `research/policies/active_policy.json`
- Active model: `research/models/20260627-mixed-winner-8task-logits-router.json`
- Active dataset: `research/datasets/20260627-mixed-winner-8task-routing.jsonl`
- Evaluation report: `research/evals/results/20260627-active-policy-eval.json`
- Tool: `tools/evaluate_active_policy.py`

## Result

`tools/evaluate_active_policy.py` now loads the active model from the registry
and evaluates it on either the active dataset or an explicitly supplied routing
dataset.

On the active dataset:

- target accuracy: 8/8
- pass@1: 8/8
- mean KL: 0.0015660431860364948
- selected workers: Qwen, Kimi, and GLM

## Decision

The promoted policy is now part of the evaluation path, not only a stored
artifact. Future benchmark reports can evaluate the active learned policy by
registry path, making policy refreshes observable without changing command-line
model paths by hand.

Next, add active-policy comparison into the broader router baseline report so
the active learned policy is shown next to strongest-worker, fastest-worker, and
rule/nearest-neighbor baselines.
