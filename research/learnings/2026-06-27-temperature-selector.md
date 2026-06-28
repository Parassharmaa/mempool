# Reward-Temperature Selector

## Result

Reward-temperature selection is now automated for the logits router. The new
selector trains a set of reward-tempered candidates, runs each candidate through
the policy refresh gate, and chooses the best promotable candidate by:

1. leave-one-out target accuracy
2. leave-one-out pass@1
3. lower leave-one-out latency regret
4. lower leave-one-out KL

Artifact:

- selection report: `research/refreshes/20260627-mixed-winner-10task-temperature-selection.json`

## Active Dataset Check

The selector was run on
`research/datasets/20260627-mixed-winner-10task-routing.jsonl` with temperatures
0.10, 0.20, and 0.50.

| Temperature | Gate decision | Warnings | LOO target accuracy | LOO pass@1 | LOO latency regret |
| ---: | --- | --- | ---: | ---: | ---: |
| 0.10 | promote | none | 0.80 | 0.90 | 518.6 ms |
| 0.20 | promote | accuracy drop, regret increase | 0.70 | 0.80 | 919.5 ms |
| 0.50 | promote | accuracy drop, regret increase | 0.70 | 0.80 | 919.5 ms |

The selector chose 0.10, matching the already active reward-tempered policy.
This confirms that the previous manual sweep is now reproducible as a gated
refresh step.

## Gate Fix

The refresh gate now uses a small floating-point tolerance for threshold
comparisons. A drop equal to the configured threshold is allowed and reported as
a warning, instead of sometimes being quarantined due to binary floating-point
rounding.

## Next Step

Use this selector when refreshing the lightweight router on new routing
datasets. The next useful data step is not another temperature sweep on the same
ten rows; it is mining more specialist-win rows and then letting the selector
choose whether the active objective temperature still holds.
