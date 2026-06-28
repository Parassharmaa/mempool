# Terminal-Bench Install-Only Preflight

## What Changed

Ran a bounded Harbor `--install-only` preflight on the first selected
Terminal-Bench pilot task, `terminal-bench/cancel-async-tasks`, and summarized
the result without reading raw logs.

Artifacts:

- `research/evals/terminal_bench_2p1_install_preflight_summary.json`
- `research/evals/terminal_bench_2p1_harness_preflight.json`

## Learning

The install-only path also failed to finalize cleanly. After interrupting at
about 131 seconds, Harbor reported one total trial with completed, errored, and
cancelled counters all nonzero. This matches the prior full oracle smoke
failure shape.

This suggests the immediate Terminal-Bench blocker is not model behavior. It is
the local Harbor/Docker job finalization path for this task.

## Decision

Do not run Terminal-Bench worker comparisons yet. Require a clean install-only
or oracle summary before spending cloud/local model calls on terminal tasks.

## Next Step

Run the next harness diagnostic against a simpler or registry-resolved task, or
inspect Harbor/Docker finalization behavior in a controlled way while keeping
raw benchmark logs under ignored `research/runs/` paths.
