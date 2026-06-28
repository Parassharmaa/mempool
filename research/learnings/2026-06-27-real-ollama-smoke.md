# 2026-06-27 - Real Ollama Smoke Evaluation

## Milestone

M1-real-worker-pool-evaluation

## Question

Can the local OpenAI-compatible worker adapter evaluate real Ollama models on
the executable smoke benchmark and produce outcome records for training?

## Method

Ran:

```bash
PYTHONPATH=src python3 tools/run_real_smoke_benchmark.py \
  --run-id 20260627-m1-real-workers \
  --output research/evals/results/20260627-m1-real-workers.json \
  --outcomes research/evals/results/20260627-m1-real-workers.jsonl
```

Workers:

- `qwen3:1.7b`
- `qwen3:4b-instruct`
- `LiquidAI/lfm2.5-1.2b-instruct:latest`

## Result

| worker | solved | pass@1 | mean latency ms | notes |
| --- | ---: | ---: | ---: | --- |
| `qwen3-1.7b` | 4/4 | 1.00 | 18589.25 | strongest pass rate but slowest |
| `qwen3-4b-instruct` | 3/4 | 0.75 | 5290.25 | failed key-normalization edge case |
| `lfm2.5-1.2b` | 1/4 | 0.25 | 2880.50 | fastest but poor code reliability |

The run produced 12 flat JSONL outcome records.

## Interpretation

The real worker evaluation path works. This is the first non-fixture signal.
Local cost is zero in this run, so the useful dimensions are pass rate, latency,
and failure mode. The early routing lesson is not simply "bigger is better":
`qwen3:1.7b` solved more than `qwen3:4b-instruct` on this tiny set, but with
much higher latency.

## Next Step

Move to M2. Convert the outcome JSONL into canonical training records with
per-task worker rewards and soft routing targets.
