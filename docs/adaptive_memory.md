# Adaptive Memory

The project should treat memory as a path toward learned capability, not only as
external retrieval.

## Principle

Repeated experience should move through stages:

1. External trace: task, route, worker outputs, verifier result, cost, latency,
   and user feedback are stored in the ledger.
2. Distilled example: useful traces are converted into clean training records.
3. Policy update: the orchestrator learns better route, workflow, verifier, and
   abstention choices from those records.
4. Lightweight adaptation: high-value repeated patterns can become adapters,
   LoRA updates, or other compact model updates.

The goal is for recurring experience to become actual learned behavior rather
than staying forever as a retrieval-time lookup.

## Fast Retraining Loop

The orchestrator should be small and cheap enough to refresh frequently. The
aspirational target is hourly retraining or adapter refresh once the data path is
stable.

Early versions can use simpler schedules:

- per-run update for the lightweight ranker
- daily adapter refresh for local fine-tunes
- manual promotion of high-confidence memories into training data

## What Changes Over Time

The system should adapt to:

- new workers entering the pool
- worker regressions or improvements
- user-specific task patterns
- repeated project conventions
- verifier evidence about routes that actually work
- cost and latency drift

## Guardrails

Frequent adaptation needs guardrails:

- keep immutable raw traces
- version every distilled dataset
- evaluate before promoting a new policy
- keep rollback points
- never train on secrets or private raw content without explicit approval
- separate user-specific memory from general model capability

## Research Bet

The novelty target is a live local orchestrator whose memory can be fused into
small policy updates. The ledger remains the source of truth, but the best
lessons should gradually migrate into the model through lightweight retraining.
