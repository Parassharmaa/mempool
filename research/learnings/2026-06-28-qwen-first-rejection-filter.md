# Qwen-First Rejection Filter

Run tag: `20260628-qwen-first-rejection-filter`

## Question

Can we reduce wasted specialist calls by running Qwen first on Qwen-resistant specialist candidates, then only spending GLM/DeepSeek/Kimi calls when Qwen fails or is slow?

The prior Qwen-resistant rank-1 run found a specialist-positive row, but Qwen still passed and won latency. The next acquisition primitive should reject Qwen-fast rows before full comparison.

## Utility Added

Added `tools/select_worker_rejection_tasks.py`.

The utility selects source tasks where a given worker:

- failed, or
- passed but exceeded `--max-pass-latency-ms`

Tests were added in `tests/test_select_worker_rejection_tasks.py`.

This makes Qwen-first acquisition explicit and reusable.

## Fresh Candidate Pool

Scanned normal `bigcode/bigcodebench` from offset `316`.

Scan result:

- scanned rows: `43`
- eligible tasks: `20`
- next offset: `359`

The eligible pool was mostly filesystem, subprocess, and general tasks, with missing-package rejects still dominated by `pandas`, `numpy`, `matplotlib`, `nltk`, and similar local environment gaps.

## Candidate Selection

The Qwen-fail contrast selector used:

- active policy registry
- rank-1 specialist filter
- Qwen-fail seeds from `research/datasets/20260628-normal-offset236-boundary-61task-routing.jsonl`
- Qwen-fast anchor seeds from the same dataset
- fresh offset316 tasks plus prior eligible pools

Selected five candidates:

- `bigcodebench-hard-BigCodeBench-327`
- `bigcodebench-hard-BigCodeBench-322`
- `bigcodebench-hard-BigCodeBench-271`
- `bigcodebench-hard-BigCodeBench-277`
- `bigcodebench-hard-BigCodeBench-328`

Only the first candidate had a strong positive Qwen-resistant score:

- `BigCodeBench-327`: Qwen-fail similarity `7.9077`, Qwen-anchor similarity `5.4452`, DeepSeek top with Kimi close second

The final two general tasks had negative scores and were useful mostly for testing the Qwen-first rejection behavior.

## Qwen-First Screen

Qwen was run first on all five candidates:

- `BigCodeBench-327`: Qwen failed, `2046 ms`
- `BigCodeBench-322`: Qwen failed, `2883 ms`
- `BigCodeBench-271`: Qwen failed, `2329 ms`
- `BigCodeBench-277`: Qwen passed, `2095 ms`
- `BigCodeBench-328`: Qwen passed, `2118 ms`

The rejection filter selected three tasks for specialist screening:

- `BigCodeBench-327`
- `BigCodeBench-322`
- `BigCodeBench-271`

It skipped two Qwen-fast rows:

- `BigCodeBench-277`
- `BigCodeBench-328`

## Specialist Screen

The three Qwen-rejected tasks were screened against GLM, DeepSeek, and Kimi.

All three were universal specialist failures:

- `BigCodeBench-327`: all specialists failed
- `BigCodeBench-322`: all specialists failed
- `BigCodeBench-271`: all specialists failed

No routing row was merge-ready for training.

## Interpretation

The Qwen-first filter worked as an acquisition-control mechanism. It avoided specialist spend on two Qwen-fast tasks and avoided top-four comparison across the whole batch.

But this run did not add a non-Qwen target. In this batch, Qwen failure predicted task hardness more than specialist opportunity.

The useful distinction is now:

- Qwen-fast rows should be skipped early.
- Qwen-fail rows should not automatically trigger specialist comparison unless the selector also has stronger evidence of specialist solvability.

## Decision

Keep the utility and the run evidence. Do not merge any rows into the routing dataset.

## Next Step

Combine Qwen-first rejection with a specialist-solvability prior:

- run Qwen first on Qwen-resistant candidates
- only run specialists when Qwen fails and the candidate is close to known specialist-positive seeds
- skip Qwen-failed rows that are closer to all-fail diagnostics than to specialist-positive rows

This should reduce both forms of waste: Qwen-fast rows and universal-failure rows.
