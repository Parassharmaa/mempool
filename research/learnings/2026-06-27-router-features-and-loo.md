# 2026-06-27 - Router Features And Leave-One-Out

## Milestone

M3-lightweight-router

## Question

Do simple prompt features improve the lightweight router, and do they generalize
under leave-one-out evaluation?

## Method

Added keyword/task-family features and a nearest-neighbor router. Regenerated the
router report:

```bash
PYTHONPATH=src python3 tools/train_router_baseline.py \
  --dataset research/datasets/20260627-real-ollama-routing.jsonl \
  --output research/datasets/20260627-router-feature-report.json
```

## Result

| policy | pass@1 | target accuracy | mean latency ms |
| --- | ---: | ---: | ---: |
| family-router | 1.00 | 0.75 | 16407.00 |
| nearest-neighbor-router | 1.00 | 1.00 | 9017.00 |
| family-router-loo | 0.75 | 0.00 | 12941.00 |
| nearest-neighbor-router-loo | 0.50 | 0.00 | 11863.00 |
| strongest-worker | 1.00 | 0.25 | 18589.25 |
| oracle-target | 1.00 | 1.00 | 9017.00 |

## Interpretation

Feature-based nearest neighbor can exactly fit this seed set, but
leave-one-out results show no reliable generalization yet. This is a useful
negative signal. The next router step should not be a neural model; it should be
more evaluated tasks.

## Next Step

Expand the real-worker smoke set before moving deeper into M3. The current
dataset is too small to justify a trained neural router.
