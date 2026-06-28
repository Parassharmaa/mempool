# Small Orchestrator Readiness Gate

Added a repeatable M5 readiness audit:

- `src/mempool/small_orchestrator_readiness.py`
- `tools/audit_small_orchestrator_readiness.py`
- `research/programs/small_orchestrator_readiness.json`

The active 23-task policy passes the worker-logits quality checks:

- 4 target workers
- 23 routing rows with at least 4 workers per task
- LOO target accuracy: 0.7826
- LOO solvable pass@1: 0.8
- LOO latency regret: 501.1 ms

It is not ready for M5 small-orchestrator training yet. The gate currently
blocks on:

- only 23 repeated routing tasks, below the 50-task threshold
- no explicit workflow-kind logits head
- no explicit abstain-probability head

Use the current gated fallback report as the teacher signal for verifier or
fallback probability. The next implementation step is to add an explicit
multi-head output contract, then keep mining repeated BigCodeBench rows until
the active dataset clears the 50-task threshold.
