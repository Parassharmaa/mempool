# 2026-06-27 - Lightweight Router Baseline

## Milestone

M3-lightweight-router

## Question

Can a tiny non-LLM router trained on the seed routing dataset beat simple
single-worker baselines on the local smoke set?

## Method

Validated the routing dataset:

```bash
PYTHONPATH=src python3 tools/validate_routing_dataset.py \
  research/datasets/20260627-real-ollama-routing.jsonl
```

Trained/evaluated a family-memorization router:

```bash
PYTHONPATH=src python3 tools/train_router_baseline.py \
  --dataset research/datasets/20260627-real-ollama-routing.jsonl \
  --output research/datasets/20260627-family-router-report.json
```

## Result

| policy | pass@1 | target accuracy | mean latency ms |
| --- | ---: | ---: | ---: |
| family-router | 1.00 | 0.75 | 16407.00 |
| strongest-worker | 1.00 | 0.25 | 18589.25 |
| fastest-worker | 0.25 | 0.25 | 2880.50 |
| oracle-target | 1.00 | 1.00 | 9017.00 |

The family router learned:

- `code_easy` -> `qwen3-4b-instruct`
- `code_text` -> `lfm2.5-1.2b`
- `code_data` -> `qwen3-1.7b`

## Interpretation

Even this tiny router improves latency over strongest-only while preserving
pass@1 on the seed set. The gap to oracle shows why more features are needed:
`code_data` contains tasks where different workers are optimal.

## Next Step

Add richer prompt/task features and evaluate leave-one-out behavior before
moving to a neural router.
