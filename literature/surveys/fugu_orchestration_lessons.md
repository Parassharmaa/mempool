# Fugu Orchestration Lessons

This note captures outside context for the internal architecture decision. It is
not project positioning.

## Useful Lessons

The Fugu technical report frames orchestration as a learned language-model
policy over a pool of stronger workers. The low-latency variant selects a single
worker per input, while the higher-quality variant composes multiple agents and
workflows for harder tasks.

Useful design lessons for `mempool`:

- treat orchestration as its own learned model, not a hand-coded router
- use a swappable worker pool so newly available models can be added without
  changing user-facing APIs
- keep a fast path that selects one worker for ordinary tasks
- reserve multi-agent composition for tasks where added latency can buy quality
- train against measured outcomes, then improve with stronger rollout feedback

## Adaptation

`mempool` should not copy a closed product shape. The open path here is a small
Qwen-style local backbone with explicit routing heads, transparent ledgers, and
auditable benchmark data.

The first neural target is:

```text
task/state text -> Qwen-small backbone -> decision hidden state -> logits heads
```

Heads:

- worker distribution
- workflow distribution
- verifier probability
- abstain/fallback probability
- later: turn action and memory-update distributions

The current linear logits router remains the baseline. It proves the outcome
dataset can train a policy, but it is not the intended final orchestrator.
