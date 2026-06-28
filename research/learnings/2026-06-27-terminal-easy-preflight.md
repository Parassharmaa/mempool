# Terminal-Bench Easy Install Preflight

## What Changed

Ran the timeout-bounded preflight wrapper against an easier Terminal-Bench 2.1
task, `terminal-bench/fix-git`, using Harbor `--install-only`.

Artifacts:

- `research/evals/terminal_bench_2p1_fix_git_install_preflight_summary.json`
- `tools/run_terminal_bench_preflight.py`

## Learning

The wrapper timed out cleanly after 300 seconds. The structured Harbor result
still reported one running trial, no completed trials, and no errored or
cancelled trials. No Harbor process or task Docker container remained afterward.

This is a different failure shape from the interrupted `cancel-async-tasks`
attempts, but it still fails the acceptance gate. The common issue is that local
Harbor jobs are not reaching a clean `complete` summary even for install-only
preflight.

## Decision

Do not spend worker calls on Terminal-Bench yet. The safe gate remains:
Terminal-Bench execution can proceed only after the wrapper emits a summary with
`process_status: exited` and `harbor_summary.status: complete`.

## Next Step

Debug Harbor locally against a toy task or registry-resolved task outside the
real benchmark content path. If the toy task completes, compare its config
against these Terminal-Bench task runs to isolate the environment finalization
issue.
