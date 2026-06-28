# Benchmark Evaluation Artifacts

This folder keeps benchmark task lists, worker-pool configs, and small
versioned benchmark-environment profiles.

The core `mempool` package intentionally has no heavy benchmark dependencies.
When a benchmark needs packages such as pandas or matplotlib, create an isolated
environment and install the relevant profile there, for example:

```bash
python3 -m venv .venv-bigcodebench
.venv-bigcodebench/bin/python -m pip install -r research/evals/bigcodebench_dependency_profile_top4.txt
PYTHONPATH=src .venv-bigcodebench/bin/python tools/scan_bigcodebench_eligible.py ...
```

Do not add benchmark-only packages to `pyproject.toml` unless the orchestrator
runtime itself needs them.

Current BigCodeBench profile evidence:

- Baseline local environment: 29 canonical-pass rows out of 148 scanned.
- Top-four profile (`pandas`, `numpy`, `matplotlib`, `requests`): 69
  canonical-pass rows out of the same 148 scanned.
- Next dependency frontier after the top-four profile: `scikit-learn`,
  `scipy`, `seaborn`, and `bs4`.

Fresh top-four acquisition artifacts:

- `bigcodebench_hard_top4_fresh_batch8_tasks.json`: first unevaluated diverse
  8-task batch.
- `bigcodebench_hard_top4_fresh_qwen_positive_tasks.json`: 3 tasks solved by
  Qwen in the one-sample mining pass.
- `bigcodebench_hard_top4_specialist_fresh_batch8_tasks.json`: second fresh
  8-task batch mined by the non-Qwen specialist pool.
- `bigcodebench_hard_top4_specialist_positive_tasks.json`: 2 specialist-positive
  tasks that became broad-pass Qwen-latency rows after repeat comparison.
- `bigcodebench_hard_top4_hard_fresh_batch8_tasks.json`: hard-strategy batch
  biased toward high-risk/network/archive/data tasks.
- `bigcodebench_hard_top4_hard_specialist_positive_tasks.json`: 3 hard-batch
  specialist positives, including two repeat-confirmed Qwen-negative Kimi
  targets.

Terminal-Bench pilot artifacts:

- `terminal_bench_2p1_pilot_plan.json`: planned metadata-only pilot ladder for
  the first agentic terminal harness check. It is a held-out evaluation track,
  not a replacement for BigCodeBench outcome collection.
- `terminal_bench_2p1_*_refreshed_summary.json`: safe refreshes of Harbor
  preflight summaries that read only `result.json` and `config.json`, never raw
  task logs.
- `terminal_bench_2p1_refreshed_readiness.json`: readiness gate over refreshed
  summaries. Worker comparisons should wait until at least one preflight has
  `process_status=exited` and `harbor_summary.status=complete`.
- `terminal_bench_2p1_fresh_oracle_readiness.json`: clean readiness evidence for
  a fresh `fix-git` install-only run plus a fresh oracle execution run.
- `terminal_bench_2p1_fix_git_oracle_fresh_trajectories.jsonl`: first
  metadata-safe Terminal-Bench trajectory row, converted from structured Harbor
  trial results and validated against the trajectory schema.
- `terminal_bench_2p1_fix_git_qwen_next_trajectories.jsonl`: first
  metadata-safe non-oracle worker trajectory row, using Terminus 2 with
  Ollama Cloud `qwen3-coder-next`.
- `terminal_bench_2p1_fix_git_oracle_vs_qwen_next_report.json`: summary-only
  comparison showing the oracle passed `fix-git` while the first Qwen Coder Next
  worker pilot completed but failed the verifier.

Terminal-Bench task instructions, oracle solutions, and verifier code should
not be copied into mempool datasets. Use metadata-only task IDs/categories for
pilot selection, then let the external harness materialize tasks inside its own
sandbox when running the benchmark.
