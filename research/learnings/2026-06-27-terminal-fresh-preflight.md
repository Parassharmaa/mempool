# Terminal-Bench Fresh Preflight

Ran a fresh Terminal-Bench `fix-git` preflight with a new Harbor job name.

Artifacts:

- `research/evals/terminal_bench_2p1_fix_git_install_fresh_summary.json`
- `research/evals/terminal_bench_2p1_fix_git_install_fresh_refreshed_summary.json`
- `research/evals/terminal_bench_2p1_fix_git_oracle_fresh_summary.json`
- `research/evals/terminal_bench_2p1_fix_git_oracle_fresh_refreshed_summary.json`
- `research/evals/terminal_bench_2p1_fresh_oracle_readiness.json`
- `research/evals/terminal_bench_2p1_fix_git_oracle_fresh_trajectories.jsonl`
- `tools/convert_harbor_job_to_terminal_trajectories.py`

Results:

- install-only preflight exited cleanly in 20.728 seconds
- oracle execution exited cleanly in 26.858 seconds
- both refreshed summaries pass the readiness gate
- the oracle trial produced a safe trajectory row with
  `task_success=true`, `verifier_passed=true`, and latency `26348.692 ms`

The trajectory converter reads only structured Harbor trial `result.json`
metadata and writes summary fields compatible with
`research/evals/terminal_bench_trajectory_schema.md`. It does not persist raw
terminal logs, task instructions, oracle solution text, or verifier code.

Decision: Terminal-Bench is ready for a tiny fixed-worker pilot on `fix-git`.
Keep it held out from router training. The next step is to run one low-cost
OpenAI-compatible/Ollama worker through the same task, convert the Harbor result
to the safe trajectory schema, and compare against the oracle smoke before
expanding to the five-task metadata pilot.
