# BigCodeBench Mini-Pilot Run

## Change

Ran the canonical-probed 3-task BigCodeBench-Hard mini-pilot through the
resumable real-worker runner and converted the 9 worker/task outcomes into
routing records.

Artifacts:

- `research/evals/results/20260627-bigcodebench-minipilot.json`
- `research/evals/results/20260627-bigcodebench-minipilot.jsonl`
- `research/datasets/20260627-bigcodebench-minipilot-routing.jsonl`
- `research/datasets/20260627-bigcodebench-minipilot-router-report.json`

## Results

Worker outcomes:

- `qwen3-1.7b`: 1/3, mean latency 81506 ms
- `qwen3-4b-instruct`: 1/3, mean latency 10833 ms
- `lfm2.5-1.2b`: 0/3, mean latency 3277 ms

Task outcomes:

- `BigCodeBench/15`: no worker passed
- `BigCodeBench/19`: qwen3-1.7b and qwen3-4b-instruct passed
- `BigCodeBench/13`: no worker passed

Router report:

- strongest-worker pass@1: 0.333, mean latency 81506 ms
- fastest-worker pass@1: 0.000, mean latency 3277 ms
- nearest-neighbor in-sample pass@1: 0.333, mean latency 4741 ms
- nearest-neighbor leave-one-out pass@1: 0.000
- oracle target pass@1: 0.333, mean latency 4741 ms

## Learning

The first external multi-task signal is useful but still too small for a
trustworthy learned router. It shows a real routing opportunity: on the one
solved task, qwen3-4b-instruct matched qwen3-1.7b accuracy while being much
faster. It also shows a failure mode: when all workers fail, the current reward
target chooses the fastest failure, which is useful for cost control but not a
capability label.

All three tasks share the same coarse family (`bigcodebench_hard`), so the
family router collapses. Routing records now include richer prompt features
such as categories, libraries, missing libraries, environment risk, and
plausibility score. These features are necessary before scaling beyond toy
families.

## Next Step

Run a larger canonical-pass external slice, or install a benchmark dependency
profile so more BigCodeBench-Hard tasks are eligible. Do not train a neural
orchestrator yet; the external dataset remains too small and too failure-heavy.
