# Orchestrated Worker Execution

Run tag: `20260628-orchestrated-worker-execution`

The trained task-level orchestrator can now drive a route-then-execute path.

## What Changed

- Added `mempool.orchestrated_executor`.
- Added `tools/run_orchestrated_prompt.py`.
- Added a dry-run mode so routing and request provenance can be tested without
  spending live worker calls.
- Added tests with a fake chat client.
- Saved a dry-run example at
  `research/evals/20260628-orchestrated-worker-execution-dry-run.json`.

## Decision

Keep the executor as the next runtime layer. It uses the existing
OpenAI-compatible worker-pool config and keeps the route decision, selected
worker, request, response, and latency in one auditable record. Live calls
should still be run intentionally, while dry-run remains the default validation
path for CI and research-loop checks.
