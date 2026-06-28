# Terminal-Bench Positioning

## Decision

Terminal-Bench 2.1 should be added to the roadmap, but after the single-step
BigCodeBench routing path.

## Why

BigCodeBench gives cheaper, cleaner worker-selection labels:

- same prompt across workers
- executable pass/fail
- direct latency and failure-mode comparison
- simple conversion into soft routing targets

Terminal-Bench measures a richer question:

- which worker should drive an interactive terminal task?
- when should the orchestrator switch or verify?
- how does the scaffold affect performance?
- how do terminal actions and feedback change worker capability?

## Plan

Use Terminal-Bench 2.1 as the first agentic harness pilot after the logits-head
orchestrator has enough single-step data. Start with a tiny subset, likely
1-3 tasks, and compare:

- best single cloud worker
- best single local worker
- static strongest-worker baseline
- logits-head router, once trained
- later, stateful orchestrator with verifier/repair actions

## Caution

Terminal-Bench evaluates model plus harness, not only raw model quality. That is
exactly why it matters for this project, but it also means we should keep
BigCodeBench as the cleaner first-stage label source.
