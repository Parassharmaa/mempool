# Terminal-Bench Summary Refresh

Added a safe refresh step for Terminal-Bench preflight summaries.

Artifacts:

- `tools/refresh_terminal_bench_preflight_summary.py`
- `src/mempool/terminal_bench.py`
- `research/evals/terminal_bench_2p1_oracle_smoke_refreshed_summary.json`
- `research/evals/terminal_bench_2p1_install_preflight_refreshed_summary.json`
- `research/evals/terminal_bench_2p1_fix_git_install_preflight_refreshed_summary.json`
- `research/evals/terminal_bench_2p1_refreshed_readiness.json`

The refresh reads only Harbor `result.json` and `config.json` metadata and keeps
`raw_log_policy=not_read`. This lets us re-check timed-out or legacy summaries
without copying raw Terminal-Bench task logs, transcripts, verifier output, task
instructions, or oracle material into mempool.

The refreshed readiness result is still blocked:

- oracle smoke: `harbor_status=interrupted_ambiguous`, missing wrapper
  `process_status`
- install-only smoke: `harbor_status=interrupted_ambiguous`, missing wrapper
  `process_status`
- easier fix-git install-only smoke: `process_status=timeout`,
  `harbor_status=running_or_stale`

Decision: keep Terminal-Bench as the held-out agentic harness, but do not start
paid/cloud worker comparisons yet. The next Terminal-Bench step is a new clean
install-only preflight with a fresh job name and enough timeout budget to exit
normally, then run the refresh and readiness tools before any model calls.
