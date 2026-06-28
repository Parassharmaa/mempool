# Reward-Tempered Router Objective

## Result

The logits router now supports training directly against a softmax over stored
worker rewards with `--target-mode reward`. A low reward temperature improved
leave-one-out behavior on the active ten-task dataset and was promoted.

Artifacts:

- model: `research/models/20260627-mixed-winner-10task-reward-t010-logits-router.json`
- report: `research/datasets/20260627-mixed-winner-10task-reward-t010-logits-router-report.json`
- refresh: `research/refreshes/20260627-mixed-winner-10task-reward-t010-refresh.json`
- active registry: `research/policies/active_policy.json`

## Sweep

All runs used `research/datasets/20260627-mixed-winner-10task-routing.jsonl`.

| Target mode | Reward temperature | LOO target accuracy | LOO pass@1 | LOO latency regret |
| --- | ---: | ---: | ---: | ---: |
| distribution | n/a | 0.70 | 0.80 | 919.5 ms |
| reward | 0.10 | 0.80 | 0.90 | 518.6 ms |
| reward | 0.20 | 0.70 | 0.80 | 919.5 ms |
| reward | 0.50 | 0.70 | 0.80 | 919.5 ms |

The `0.10` run passed the same refresh gate with no warnings:

- minimum LOO accuracy: 0.70
- maximum LOO accuracy drop: 0.10
- maximum LOO latency regret: 1000 ms
- maximum LOO latency regret increase: 500 ms

## Interpretation

The existing target distribution already came from reward, but training against
a colder reward softmax made the boundary less ambiguous for leave-one-out
generalization. This improved both target accuracy and pass@1 while cutting
mean latency regret by 400.9 ms relative to the previous active policy.

The held-out broad-pass latency diagnostic stayed healthy after promotion:

- target accuracy: 1.0
- pass@1: 1.0
- mean latency regret: 0.0 ms

## Next Step

Keep the reward-tempered objective as the active lightweight-router path. The
next experiment should make the temperature a gateable hyperparameter rather
than a manual sweep, then collect more specialist-win rows so the active model
is not mostly learning when to select Qwen.
