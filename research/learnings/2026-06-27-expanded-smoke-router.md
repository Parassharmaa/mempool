# 2026-06-27 - Expanded Smoke Router

## Milestone

M3-lightweight-router

## Question

Does expanding the real Ollama smoke benchmark from 4 to 10 tasks improve the
router signal enough to justify moving toward a neural router?

## Method

Expanded `research/evals/smoke_code_tasks.json` to 10 executable Python tasks.

Ran real workers:

```bash
PYTHONPATH=src python3 tools/run_real_smoke_benchmark.py \
  --run-id 20260627-m3-expanded-real-workers \
  --output research/evals/results/20260627-m3-expanded-real-workers.json \
  --outcomes research/evals/results/20260627-m3-expanded-real-workers.jsonl
```

Built routing records and evaluated routers:

```bash
PYTHONPATH=src python3 tools/build_routing_dataset.py \
  --input research/evals/results/20260627-m3-expanded-real-workers.jsonl \
  --output research/datasets/20260627-expanded-ollama-routing.jsonl

PYTHONPATH=src python3 tools/train_router_baseline.py \
  --dataset research/datasets/20260627-expanded-ollama-routing.jsonl \
  --output research/datasets/20260627-expanded-router-report.json
```

## Result

Real worker pass rates:

| worker | solved | pass@1 | mean latency ms |
| --- | ---: | ---: | ---: |
| `qwen3-1.7b` | 7/10 | 0.70 | 18875.60 |
| `qwen3-4b-instruct` | 8/10 | 0.80 | 2186.10 |
| `lfm2.5-1.2b` | 3/10 | 0.30 | 689.80 |

Router results:

| policy | pass@1 | target accuracy | mean latency ms |
| --- | ---: | ---: | ---: |
| family-router | 0.50 | 0.60 | 895.50 |
| nearest-neighbor-router | 0.90 | 1.00 | 3237.60 |
| family-router-loo | 0.50 | 0.40 | 1104.00 |
| nearest-neighbor-router-loo | 0.60 | 0.40 | 1699.30 |
| strongest-worker | 0.80 | 0.50 | 2186.10 |
| fastest-worker | 0.30 | 0.40 | 689.80 |
| oracle-target | 0.90 | 1.00 | 3237.60 |

## Interpretation

The expanded benchmark gives better evidence. In-sample nearest neighbor reaches
oracle, but leave-one-out still underperforms the strongest-worker baseline on
pass@1. This means the current router is not ready to justify neural training.

The strongest single worker changed from `qwen3-1.7b` on the 4-task set to
`qwen3-4b-instruct` on the 10-task set, which confirms that tiny evals can
mislead.

## Next Step

Before M4, improve the real-run harness with resumability/progress logging and
add more tasks or a small external benchmark subset. A neural router should wait
until leave-one-out or holdout routing beats strongest-worker on at least one
meaningful metric without unacceptable pass-rate loss.
