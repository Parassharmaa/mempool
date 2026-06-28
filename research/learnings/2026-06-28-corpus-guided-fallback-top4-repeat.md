# Corpus-Guided Fallback Top-Four Repeat

Run tag: `20260628-corpus-guided-fallback-top4-repeat`

## Question

Does the corpus-guided fallback acquisition batch produce fresh fallback
positives, cheap second-attempt rescues, or useful worker-selection evidence?

## Run

Evaluated the eight selected BigCodeBench-Hard tasks from
`research/evals/bigcodebench_hard_corpus_guided_fallback_batch8_tasks.json`
against the top-four Ollama Cloud worker pool with two samples per worker/task.

Artifacts:

- raw summary: `research/evals/results/20260628-corpus-guided-fallback-top4-repeat.json`
- outcome rows: `research/evals/results/20260628-corpus-guided-fallback-top4-repeat.jsonl`
- repeated summary: `research/evals/results/20260628-corpus-guided-fallback-top4-repeat-summary.json`
- routing dataset: `research/datasets/20260628-corpus-guided-fallback-top4-repeat-routing.jsonl`
- fallback mining report: `research/evals/20260628-corpus-guided-fallback-top4-repeat-fallback-report.json`
- 38-task refresh selection: `research/evals/20260628-live-plus-corpus-guided-38task-selection.json`

## Result

The batch produced 64 outcome rows and 8 routing records.

Six tasks were solvable by at least one worker:

- `BigCodeBench/281`
- `BigCodeBench/339`
- `BigCodeBench/539`
- `BigCodeBench/671`
- `BigCodeBench/673`
- `BigCodeBench/675`

Two tasks were universal failures:

- `BigCodeBench/322`
- `BigCodeBench/327`

Worker pass rates across the eight-task slice:

- GLM: 12/16 samples, pass rate 0.7500
- Kimi: 10/16 samples, pass rate 0.6250
- Qwen: 10/16 samples, pass rate 0.6250
- DeepSeek: 6/16 samples, pass rate 0.3750

Latency-adjusted targets in the routing dataset:

- Qwen: `BigCodeBench/281`, `322`, `327`
- GLM: `BigCodeBench/339`
- Kimi: `BigCodeBench/539`, `671`, `673`, `675`

## Active Policy Diagnostic

The current active logits router on this fresh slice:

- target accuracy: 0.6250
- pass@1: 0.7500
- solvable pass@1: 1.0000
- mean latency regret: 6525.625 ms

So the active policy solved every solvable task, but often routed to slower
passing workers.

## Fallback Mining

Mining active-router top-fail cases produced two fallback opportunities, both
hard negatives:

- `BigCodeBench/322`
- `BigCodeBench/327`

There were no useful-any or useful-second fallback positives. This means the
corpus-guided selector improved worker-boundary evidence, but did not add the
fallback-positive labels needed by the value head.

## Refresh Gate

Merged the new 8-task dataset with the live-augmented 30-task dataset into
`research/datasets/20260628-live-plus-corpus-guided-38task-routing.jsonl` and
ran a temperature sweep initialized from the active model against the
probe-gated operational reference.

All raw retrains were quarantined:

- temperature 0.05: target accuracy 0.7368, pass@1 0.7895, solvable pass@1
  0.9091, latency regret 1903.3 ms
- temperature 0.10: target accuracy 0.7105, pass@1 0.7895, solvable pass@1
  0.9091, latency regret 2173.8 ms
- temperature 0.20: target accuracy 0.6842, pass@1 0.7632, solvable pass@1
  0.8788, latency regret 5227.6 ms

The operational probe-gated bar remains much stricter: target accuracy 0.8333
and latency regret 251.4 ms.

## Decision

Keep:

- new repeated top-four outcome dataset
- new routing records
- new 38-task merged stress-test dataset
- refresh-gate quarantine result

Do not promote:

- raw 38-task logits-router refresh
- value-head/fallback-policy update from this batch

## Next

The next acquisition should explicitly target active-router first-attempt
failures that are solvable by a non-top worker. This batch mostly says the
active router needs latency calibration on solvable rows, not that the fallback
value head has enough new positives.
