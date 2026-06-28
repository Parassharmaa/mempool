# Conditional Fallback Prototype

## Question

Can a verifier-style fallback layer rescue solvable tasks that the active logits
router sends to a fast failing worker?

## Setup

Added an evaluation-only conditional policy:

1. Rank workers by the active logits-router distribution.
2. Try the first worker.
3. If the recorded verifier result fails, try the next ranked worker.
4. Stop on the first passing worker or after `max_attempts`.

This is not yet a live rollout policy. It is an offline simulation over measured
outcomes that tells us whether verifier-guided fallback is worth implementing.

## Result

On the active 23-task dataset with `max_attempts=2`:

- base logits router: 18/23 solved, solvable pass@1 0.90
- conditional fallback: 19/23 solved, solvable pass@1 0.95
- mean latency increased from about 4.4s to 6.7s

On regression slices:

- base logits router passes the DeepSeek `763` slice but fails the GLM `526`
  slice
- conditional fallback with `max_attempts=2` passes both slices

The GLM slice is rescued by the second-ranked DeepSeek attempt, not by selecting
GLM directly. This matters: the fallback improves solved-task coverage, but it
does not learn the true GLM target. It is a verifier/repair behavior, not a
better first-route classifier.

## Interpretation

Verifier-guided fallback is a real path for the orchestrator. It can prevent the
most harmful version of fastest-failure routing while keeping the first-route
logits model small.

The tradeoff is latency. On all-fail rows, fallback spends extra attempts and
can substantially increase latency. A production policy should learn when to
fallback, not fallback blindly on every failed first attempt.

## Next Step

Train or gate a conditional verifier/abstain head that predicts whether the
router should spend a second attempt. Evaluate that head against both regression
slices and mean latency regret before making it active.
