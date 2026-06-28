# Latency-Safe Matched Controls

Added a matched-control acquisition selector for the latency-safe head:

- `tools/select_latency_safe_matched_controls.py`
- `tests/test_select_latency_safe_matched_controls.py`

Purpose:

The latency-safe head could memorize current labels, but leave-one-out precision
was only 0.625. The missing data is not more random tasks; it is matched
controls:

- candidate rows likely to be all-pass latency rows,
- nearby unsafe controls where at least one worker fails.

Generated artifacts:

- selected tasks:
  `research/evals/20260628-latency-safe-matched-controls-tasks.json`
- report:
  `research/evals/20260628-latency-safe-matched-controls-report.json`
- execution manifest:
  `research/evals/20260628-latency-safe-matched-controls-manifest.json`

Selection summary:

- seed dataset:
  `research/datasets/20260628-normal-offset16-contrast-29task-weight0p25-routing.jsonl`
- safe seeds: 13
- unsafe seeds: 16
- fresh candidates after exclusions: 60
- selected tasks: 8
- selected latency-safe candidates: 4
- selected unsafe controls: 4

Selected task ids:

- `bigcodebench-hard-BigCodeBench-320`
- `bigcodebench-hard-BigCodeBench-391`
- `bigcodebench-hard-BigCodeBench-305`
- `bigcodebench-hard-BigCodeBench-127`
- `bigcodebench-hard-BigCodeBench-384`
- `bigcodebench-hard-BigCodeBench-325`
- `bigcodebench-hard-BigCodeBench-547`
- `bigcodebench-hard-BigCodeBench-336`

Next live run:

```bash
PYTHONPATH=src python3 tools/run_real_smoke_benchmark.py --config research/evals/ollama_cloud_worker_pool_top4.json --tasks research/evals/20260628-latency-safe-matched-controls-tasks.json --repeat-count 2 --eval-timeout-seconds 40 --run-id 20260628-latency-safe-matched-controls-top4 --output research/evals/results/20260628-latency-safe-matched-controls-top4.json --outcomes research/evals/results/20260628-latency-safe-matched-controls-top4.jsonl --resume --progress --quiet
```

Learning:

This moves the latency-safe condition from an oracle allowlist toward a
measurable training loop. The selected batch is intentionally balanced, so after
the live run we can rebuild the latency-safe labels and test whether matched
controls improve leave-one-out precision.
