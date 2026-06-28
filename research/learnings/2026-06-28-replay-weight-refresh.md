# Replay-Weighted Refresh

Added conservative replay weights for logits-router refresh experiments:

- `training_weight` is now an optional routing-record field consumed by
  `train_logits_router`.
- `tools/annotate_routing_dataset_weights.py` creates weighted candidate
  datasets without changing canonical outcome records.
- Tests:
  - `tests/test_logits_router.py`
  - `tests/test_annotate_routing_dataset_weights.py`

Experiment:

- base dataset:
  `research/datasets/20260627-mixed-winner-23task-heldout-hard-routing.jsonl`
- new clean broad-pass latency slice:
  `research/datasets/20260628-normal-offset16-contrast-routing-merge-ready.jsonl`
- merged candidate:
  `research/datasets/20260628-normal-offset16-contrast-29task-routing.jsonl`
- replay curve:
  `research/evals/20260628-replay-weight-refresh-curve.json`
- promotion profile: `preserve_accuracy`
- reward temperature: `0.05`

Results:

| new-row weight | decision | target accuracy | pass@1 | solvable pass@1 | latency regret |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.0 | quarantine | 0.5517 | 0.6897 | 0.7692 | 783.2 ms |
| 0.5 | quarantine | 0.5517 | 0.6897 | 0.7692 | 783.2 ms |
| 0.25 | quarantine | 0.5862 | 0.7241 | 0.8077 | 755.0 ms |
| 0.1 | quarantine | 0.5862 | 0.7241 | 0.8077 | 755.0 ms |

Baseline:

- target accuracy: 0.7826
- pass@1: 0.6957
- solvable pass@1: 0.8
- latency regret: 501.1 ms

Learning:

Replay weighting reduces the small-batch drift enough to improve pass@1 and
solvable pass@1, but it still fails the preserve-accuracy gate because target
accuracy and latency regret remain too weak. Keep the code path because it is a
useful refresh guardrail primitive, but do not promote any replay-weighted
candidate from this slice.

Next step:

Use replay weights as a guardrail in future refresh cycles, but the router still
needs either better features/capacity or matched control acquisition before it
can absorb broad-pass latency rows safely.
