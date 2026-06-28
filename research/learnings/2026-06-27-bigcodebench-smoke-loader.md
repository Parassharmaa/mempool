# BigCodeBench-Hard Smoke Loader

## Change

Added a dependency-free BigCodeBench-Hard materializer that fetches rows from
the Hugging Face dataset-server API or reads local JSON/JSONL exports, then
normalizes them into the existing `SmokeCodeTask` schema.

The materialized smoke file is:

- `research/evals/bigcodebench_hard_smoke_tasks.json`

The real-worker runner can now select a task file with `--tasks`, and external
evaluations can use a longer `--eval-timeout-seconds` value.

## First Signal

Ran the first materialized BigCodeBench-Hard task against the current three
Ollama workers:

- task: `bigcodebench-hard-BigCodeBench-13`
- qwen3-1.7b: failed, 114472 ms
- qwen3-4b-instruct: failed, 17529 ms
- lfm2.5-1.2b: failed, 4983 ms

The canonical BigCodeBench code prompt plus canonical solution passed under the
same adapter, so this was a worker-performance failure rather than an empty-test
or broken-runner false signal.

## Router Dataset

The external outcome was converted into:

- `research/datasets/20260627-bigcodebench-hard-smoke-1-routing.jsonl`
- `research/datasets/20260627-bigcodebench-hard-smoke-1-router-report.json`

Because all workers failed, the reward target collapses to the fastest
least-bad worker. This is useful as a failure/latency datapoint, but not enough
to train or validate a meaningful router.

## Learning

The first hard task stresses exact exception-message behavior and subprocess/FTP
library use. It is a good harness test, but it is a poor first scaling target
for the local pool because it produced zero pass@1 across workers and one slow
generation over 100 seconds.

Before running all 10 materialized tasks, select or stratify a smaller first
external slice that includes at least some solvable tasks. Otherwise the router
will mostly learn latency under universal failure rather than task-dependent
competence.

## Next Step

Add a smoke-task selector/report that can probe canonical tests and summarize
prompt/library characteristics, then choose a 3-task external mini-pilot before
the full 10-task smoke run.
