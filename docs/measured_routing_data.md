# Measured Routing Data

This project trains the orchestrator from measured routing data, not only from
public benchmark leaderboards.

## Why Public Scores Are Not Enough

Public benchmark scores are useful for choosing candidate workers, but they are
too coarse to train a router.

A leaderboard usually answers:

```text
worker A solved 62%
worker B solved 58%
worker C solved 51%
```

The orchestrator needs a different object:

```text
task X:
  worker A: pass/pass, 2400 ms
  worker B: fail/fail, 9000 ms
  worker C: pass/fail, 3200 ms
```

That per-task structure lets the policy learn when to route to a specialist,
when to prefer the fastest broad-pass worker, and when to abstain, verify,
repair, or escalate.

Public benchmark data is also usually mismatched to the local setting:

- model versions and hosted endpoints drift
- prompts and scaffolds differ
- sampling settings differ
- dependency environments differ
- retry policies differ
- latency and cost differ by provider
- per-task repeatability is often unavailable
- failure modes are usually not normalized

For `mempool`, the training question is:

```text
Given this task, this worker pool, this scaffold, and this environment,
which route maximizes expected reward?
```

That requires measured local outcomes.

## Two Dataset Tracks

### BigCodeBench Routing Records

BigCodeBench rows are the current training source.

They are mostly single-step code-generation examples:

```text
task prompt -> worker -> generated code -> executable tests -> reward
```

This makes them cheap and clean enough for the first router. A routing record
groups repeated outcomes for one task across workers and stores:

- task features
- worker pass/fail scores
- latency and cost estimates
- failure mode
- per-worker reward
- soft target distribution
- hard target worker

These rows train the lightweight logits router today and will later train the
worker-distribution head of the small orchestrator.

### Terminal-Bench Trajectories

Terminal-Bench rows are held-out agentic evaluation data for now.

They are multi-step terminal trajectories:

```text
task id -> agent scaffold -> worker -> terminal actions/file edits/tests ->
verifier reward
```

Terminal-Bench measures model plus scaffold behavior, so it should not be mixed
directly into the BigCodeBench routing dataset. It is used to answer later
orchestration questions:

- is the initial worker choice good for terminal tasks?
- does the policy need state/history features?
- when should the orchestrator switch workers?
- when should it call a verifier or repair?
- when should it stop or abstain?

Terminal-Bench artifacts must remain metadata-safe: no task instructions,
oracle solutions, verifier code, raw stdout/stderr, or full terminal transcripts
should be copied into `mempool` datasets.

## Training Ladder

1. Collect repeated BigCodeBench outcomes across the worker pool.
2. Convert outcomes into routing records with empirical rewards.
3. Train and gate a lightweight logits router.
4. Grow the stable routing set to at least 50 tasks with diverse target workers.
5. Train a small multi-head orchestrator:
   - worker distribution
   - workflow kind
   - verifier probability
   - abstain probability
6. Use Terminal-Bench as a held-out agentic benchmark before allowing any
   sanitized trajectory-derived training.
7. Add adaptive memory refresh only after the measured-data path supports
   rollback and evaluation.

## Current State

The active trainable routing dataset is:

```text
research/datasets/20260627-mixed-winner-23task-heldout-hard-routing.jsonl
```

It contains 23 stable BigCodeBench routing tasks across four target workers.
This is enough to train the current lightweight logits router, but not enough to
fine-tune the small orchestrator. The readiness gate requires at least 50 stable
tasks before Milestone 5 training.

The first Terminal-Bench held-out trajectory pair is:

```text
research/evals/terminal_bench_2p1_fix_git_oracle_fresh_trajectories.jsonl
research/evals/terminal_bench_2p1_fix_git_qwen_next_trajectories.jsonl
```

This proves the agentic harness path, but those rows remain evaluation-only
until a later explicit training decision.
