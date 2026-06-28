# Non-Qwen Pressure Live Screen

Run tag: `20260628-nonqwen-pressure-live-screen`

## Question

Does the `nonqwen-pressure` acquisition source produce stable non-Qwen
specialist labels after live screening and top-four repeat comparison?

## One-Sample Non-Qwen Screen

Manifest:

- `research/evals/20260628-nonqwen-pressure-top3-screen-manifest.json`

Ran:

- `6` tasks
- `3` non-Qwen workers: GLM, DeepSeek, Kimi
- `18` total calls

Artifacts:

- `research/evals/results/20260628-nonqwen-pressure-top3-screen.jsonl`
- `research/evals/results/20260628-nonqwen-pressure-top3-screen-summary.json`
- `research/evals/results/20260628-nonqwen-pressure-top3-screen-audit.json`
- `research/datasets/20260628-nonqwen-pressure-top3-screen-routing.jsonl`
- `research/datasets/20260628-nonqwen-pressure-top3-screen-merge-ready-routing.jsonl`

Result:

- `BigCodeBench-15`: one-shot DeepSeek positive
- `BigCodeBench-260`: one-shot broad positive across GLM, DeepSeek, and Kimi
- `BigCodeBench-6`, `130`, `131`, `288`: all-fail rows

The outcome audit passed, and the merge-ready filter kept two positive rows
from the non-Qwen-only screen. Because Qwen was intentionally absent, these
rows were not used as final router-training evidence.

## Top-Four Repeat Comparison

To make the labels comparable with active router evidence, the two positive
tasks were rerun across the top-four pool with two samples per worker.

Artifacts:

- `research/evals/20260628-nonqwen-pressure-positive-repeat-tasks.json`
- `research/evals/results/20260628-nonqwen-pressure-positive-repeat.jsonl`
- `research/evals/results/20260628-nonqwen-pressure-positive-repeat-summary.json`
- `research/evals/results/20260628-nonqwen-pressure-positive-repeat-audit.json`
- `research/datasets/20260628-nonqwen-pressure-positive-repeat-routing.jsonl`
- `research/datasets/20260628-nonqwen-pressure-positive-repeat-merge-ready-routing.jsonl`

Result:

- `BigCodeBench-15`: universal failure on repeat, including DeepSeek `0/2`
- `BigCodeBench-260`: all four workers passed `2/2`
- `BigCodeBench-260` target: Qwen by latency, mean `4026.5 ms`

The merge-ready filter kept only `BigCodeBench-260`. This is useful
broad-pass latency evidence, not a non-Qwen specialist win.

## Evaluation

Full suite:

- `PYTHONPATH=src python3 -m unittest discover -s tests`
- result: `223` tests passed

Research-loop gate:

- `PYTHONPATH=src python3 tools/research_loop.py evaluate --tag 20260628-nonqwen-pressure-live-screen`
- result: `pass`
- score: `1.0`

## Decision

Keep the artifacts and the screen-to-repeat workflow. Do not promote or retrain
from this run alone:

- the only repeat-merge-ready row is a Qwen latency target
- the apparent DeepSeek specialist positive did not reproduce
- the active policy should remain unchanged

Next acquisition should keep searching for stable non-Qwen specialist wins or
use the all-fail rows as negative source-selection evidence.
