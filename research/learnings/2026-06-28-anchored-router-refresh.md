# Anchored Router Refresh

Added warm-start support for logits-router refresh experiments:

- `LogitsRouter.initialize(..., initial_router=...)` copies matching
  worker/feature weights from an existing router.
- `train_logits_router(..., initial_router=...)` starts training from that
  anchor.
- `tools/select_logits_router_temperature.py --initial-model ...` records and
  uses the anchor during candidate training and leave-one-out evaluation.

Experiment:

- baseline model:
  `research/models/20260627-mixed-winner-23task-heldout-hard-reward-t0p05-logits-router.json`
- candidate dataset:
  `research/datasets/20260628-normal-offset16-contrast-29task-weight0p25-routing.jsonl`
- curve report:
  `research/evals/20260628-anchored-refresh-curve.json`
- promotion profile: `preserve_accuracy`

Best anchored candidate:

- run:
  `research/datasets/20260628-normal-offset16-contrast-29task-weight0p25-anchored-e50-lr1e4-temperature-selection.json`
- new-row weight: 0.25
- epochs: 50
- learning rate: 0.0001
- reward temperature: 0.05
- decision: quarantine

Metrics vs active baseline:

| metric | active baseline | best anchored candidate |
| --- | ---: | ---: |
| target accuracy | 0.7826 | 0.6897 |
| pass@1 | 0.6957 | 0.8276 |
| solvable pass@1 | 0.8000 | 0.9231 |
| latency regret | 501.1 ms | 744.9 ms |

Learning:

Warm-starting from the active router is clearly better than zero-initialized
refresh for pass retention. The same candidate that previously reached only
0.7241 pass@1 can reach 0.8276 pass@1 when anchored. However, it still fails
the preserve-accuracy gate because it misses too many hard targets and routes
to slower successful workers.

Keep the active 23-task policy unchanged. The next useful refresh primitive is
not a larger generic warm start; it should add latency-aware anchoring or a
two-objective gate that can distinguish "safe all-pass latency optimization"
from "specialist target learning."
