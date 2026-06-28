# Gated Fallback Threshold

## Question

Can the conditional verifier/fallback behavior be made selective enough to
recover solvable misses without paying the full latency cost of always trying a
second worker?

## Setup

Added a gated fallback evaluator. After the first ranked worker fails, it only
tries the second ranked worker when the active router's top-vs-second
probability margin is below a threshold.

Evaluated margins: 0.02, 0.05, 0.10, 0.15, 0.20.

## Result

On the active 23-task dataset:

- base logits router: 18/23 solved, solvable pass@1 0.90, mean latency 4410 ms
- always fallback max-2: 19/23 solved, solvable pass@1 0.95, mean latency
  6703 ms
- gated fallback margin 0.10: 19/23 solved, solvable pass@1 0.95, mean latency
  4887 ms

The 0.10 threshold takes 2 fallbacks out of 5 first-failure opportunities. It
passes both regression slices:

- DeepSeek `763`: pass without fallback
- GLM `526`: pass with one fallback, rescued by DeepSeek

## Interpretation

Router uncertainty is a useful first verifier/abstain feature. A small margin
between the first and second worker is enough to identify cases where spending a
second attempt can improve solvability.

The GLM regression is still not fixed as a first-route target: the fallback
solves it via DeepSeek. This is acceptable for a verifier/repair behavior, but
not as evidence that the router has learned GLM selection.

## Next Step

Promote gated fallback only as a conditional workflow mode, not as a replacement
for the first-route logits router. The next trainable orchestrator head should
emit both worker logits and a fallback/verify logit, trained against this
margin-gated behavior and later replaced by learned features.
