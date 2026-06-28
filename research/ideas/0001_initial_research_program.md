# Initial Research Program

## Goal

Build an open, auditable coordinator that learns how to route and compose work
across a pool of workers under explicit constraints.

## First Milestone

Create a minimal local harness:

- Task schema.
- Worker pool interface.
- Adapter contract for top-tier hosted models, open-weight models, and local
  specialists.
- Workflow policy interface.
- JSONL decision ledger.
- Simple evaluator for answer quality, cost, latency, and failure mode.

## First Experiments

1. Rule baseline: route by task family.
2. Learned classifier: route by task embedding and constraints.
3. Compare workflow: sample two workers and verify.
4. Conditional verifier: verify only when uncertainty is high.
5. Abstention baseline: detect tasks outside the pool's capability.
6. Pool upgrade test: add a new strong worker by adapter only and measure whether
   the coordinator benefits without code changes.

## First Benchmark

Use BigCodeBench-Hard in instruct mode as the first external benchmark target.
Start with a 10-task smoke set, then a 30-50 task pilot, then the full hard
subset if the harness is stable.

## Non-Goals For Now

- No hidden production service.
- No benchmark claims without reproducible evals.
- No dependency on a single external provider or framework.
- No open-ended autonomous edits outside bounded experiment scopes.
