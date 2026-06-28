# 2026-06-27 - Routing Dataset Seed

## Milestone

M2-outcome-dataset

## Question

Can real worker outcome records be converted into grouped router-training
records with soft target distributions?

## Method

Ran:

```bash
PYTHONPATH=src python3 tools/build_routing_dataset.py \
  --input research/evals/results/20260627-m1-real-workers.jsonl \
  --output research/datasets/20260627-real-ollama-routing.jsonl
```

Input:

- 12 outcome records
- 4 smoke tasks
- 3 real Ollama workers

## Result

Produced 4 routing records. Targets:

| task | target worker | note |
| --- | --- | --- |
| `smoke-add-numbers` | `qwen3-4b-instruct` | both Qwen workers passed; latency slightly favored 4B instruct |
| `smoke-reverse-words` | `lfm2.5-1.2b` | all passed; latency favored the fastest worker |
| `smoke-normalize-records` | `qwen3-1.7b` | only this worker passed |
| `smoke-top-k-frequent` | `qwen3-4b-instruct` | both Qwen workers passed; latency favored 4B instruct |

## Interpretation

The reward-to-soft-target path works. The tiny dataset already demonstrates why
soft labels are useful: several tasks have multiple passing workers, so the
training target should preserve tradeoffs instead of collapsing to a brittle
single label.

## Next Step

Add dataset validation checks and then train the first lightweight router
baseline against this seed format.
