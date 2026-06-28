# BigCodeBench Eligible 8-Task Run

## Change

Ran the 8-task canonical-pass BigCodeBench-Hard eligible pool across the current
Ollama worker pool and rebuilt the external routing dataset.

Artifacts:

- `research/evals/results/20260627-bigcodebench-eligible-8.json`
- `research/evals/results/20260627-bigcodebench-eligible-8.jsonl`
- `research/datasets/20260627-bigcodebench-eligible-8-routing.jsonl`
- `research/datasets/20260627-bigcodebench-eligible-8-router-report.json`

## Results

Worker outcomes:

- `qwen3-1.7b`: 1/8, mean latency 69781 ms
- `qwen3-4b-instruct`: 1/8, mean latency 8621 ms
- `lfm2.5-1.2b`: 0/8, mean latency 2411 ms

Only `BigCodeBench/19` passed, and both Qwen workers solved it. The larger
Qwen worker matched the smaller Qwen worker on pass count while being much
faster across the eligible pool.

Router report:

- strongest-worker: qwen3-4b-instruct, pass@1 0.125, mean latency 8621 ms
- fastest-worker: lfm2.5-1.2b, pass@1 0.000, mean latency 2411 ms
- family-router LOO: pass@1 0.000
- nearest-neighbor LOO: pass@1 0.000
- oracle target: pass@1 0.125, mean latency 3003 ms

## Baseline Fix

The strongest-worker baseline originally broke pass-count ties by incidental
record order. It now breaks ties by lower mean latency, so qwen3-4b-instruct is
correctly selected over qwen3-1.7b when both solve the same number of tasks.

## Learning

The eligible 8-task slice is still failure-heavy. It proves the external
evaluation and routing-data path, but it does not yet provide enough positive
variation for a reliable learned router or a neural orchestrator. Most current
routing targets are fastest-failure labels rather than capability labels.

The useful signal is about worker economics: qwen3-4b-instruct dominates
qwen3-1.7b on this slice because it preserves the only observed pass while
cutting mean latency by roughly 8x. The lightweight worker remains useful as a
cheap negative/fast-failure baseline, but not as a solver.

## Next Step

Before training a small orchestrator, collect more positive external outcomes.
The best next experiment is either:

- scan more standard-library BigCodeBench-Hard rows past offset 44, or
- add an isolated benchmark dependency profile so data-science and web tasks
can be evaluated fairly.
