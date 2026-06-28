# Orchestration Research Loop

This program defines the current autonomous research loop for `mempool`.

## Objective

Improve the coordinator so it can select useful workflow shapes over a worker
pool while preserving transparency, cost awareness, and reproducibility.

For the long-term trainable orchestrator path, follow
`research/programs/trainable_orchestrator_build_plan.md` and keep
`research/programs/milestones.json` updated.

## Fixed Evaluation For Now

The current local evaluation is a harness-health gate:

- unit tests pass
- demo planning runs
- a ledger event is written and parseable
- the selected plan uses the expected workflow fields

This is not the final benchmark. It is the safety gate that must pass before any
expensive benchmark run.

## Near-Term Research Tasks

1. Add a benchmark adapter interface. Done.
2. Add a local executable smoke benchmark. Done.
3. Evaluate real Ollama workers on the smoke benchmark.
4. Add worker result schema for generated code.
5. Add cost and latency accounting to workflow execution.
6. Add a BigCodeBench-Hard smoke-set loader.
7. Add a conditional-verifier policy.
8. Compare learned routing against single-worker baselines.

## Keep Criteria

Keep a change when it does at least one of these:

- improves benchmark pass rate
- improves cost per solved task without hurting pass rate
- improves latency without hurting pass rate
- adds required provenance or resumability
- simplifies the system without hurting measured behavior

## Discard Criteria

Discard a change when it:

- weakens provenance
- makes benchmark results harder to reproduce
- hard-codes a provider or model into coordinator logic
- improves one demo path while breaking general worker adapters
- adds complexity without a measurable reason

## Current Metric Order

1. Tests and local harness health must pass.
2. Smoke benchmark pass@1 once available.
3. Cost per solved task.
4. Latency per solved task.
5. Simplicity and inspectability.
