# Terminal-Bench Worker Pilot

Ran the first non-oracle Terminal-Bench worker pilot on `fix-git`.

Artifacts:

- `tools/run_terminal_bench_preflight.py`
- `tools/compare_terminal_bench_trajectories.py`
- `research/evals/terminal_bench_2p1_fix_git_qwen_next_summary.json`
- `research/evals/terminal_bench_2p1_fix_git_qwen_next_refreshed_summary.json`
- `research/evals/terminal_bench_2p1_qwen_next_readiness.json`
- `research/evals/terminal_bench_2p1_fix_git_qwen_next_trajectories.jsonl`
- `research/evals/terminal_bench_2p1_fix_git_oracle_vs_qwen_next_report.json`

Setup:

- agent: Harbor `terminus-2`
- worker: Ollama Cloud `qwen3-coder-next` via OpenAI-compatible endpoint
- task: Terminal-Bench 2.1 `fix-git`
- max turns: 5
- content policy: no raw terminal logs, task instructions, verifier code, or
  oracle solution text copied into mempool artifacts

Results:

- Harbor worker job exited cleanly in `69.91` seconds
- readiness gate passed with `process_status=exited` and
  `harbor_summary.status=complete`
- safe trajectory conversion produced one valid row
- verifier reward was `0.0`, so the worker trajectory has
  `task_success=false`
- oracle baseline on the same task has `task_success=true`

The comparison report is summary-only:

- fixed oracle: success rate `1.0`, mean latency `26348.692 ms`
- fixed Qwen Coder Next worker: success rate `0.0`, mean latency
  `63061.673 ms`

Decision: this is a useful first held-out agentic trajectory signal, even though
it is negative. Keep Terminal-Bench rows separate from BigCodeBench router
training for now. The next step is to try one stronger or more terminal-native
worker/scaffold on `fix-git`, or expand only after one non-oracle worker
achieves a verifier pass.
