# Fallback Logit Head Prototype

## Question

Can the conditional fallback decision become a small learned workflow-action
head instead of a fixed top-two router margin rule?

## Change

Added a trainable fallback classifier in `src/mempool/fallback_head.py` and a
training/selection CLI in `tools/train_fallback_head.py`.

The head is a logistic policy over:

- active router confidence features
- top and second worker probabilities
- top-two margin
- router entropy
- top/second worker identity
- existing task features

It only controls the decision after the first ranked worker has failed: whether
to spend one more attempt on the second ranked worker.

## Result

Training directly on actual second-attempt rescue labels was too sparse. The
active dataset has only one useful second-attempt rescue under the current
router, so the head overfit and failed the fresh `BigCodeBench/526` regression
slice.

Training against the selected `0.10` margin gate as a teacher produced a
selectable model artifact:

- `research/models/20260627-fallback-logit-head.json`
- `research/refreshes/20260627-fallback-logit-head-selection.json`

The selected learned threshold is `0.001`.

On the active 23-task dataset:

- solved: 19/23
- pass@1: 0.8261
- solvable pass@1: 0.95
- fallbacks taken: 4 of 5 first-failure opportunities
- mean latency: 6154.5 ms
- mean latency regret: 2015.9 ms

The learned head passes both regression slices, but it is worse than the
explicit margin gate, which gets the same solved count with only 2 fallbacks and
748.3 ms mean latency regret.

## Learning

The architecture direction is right: the orchestrator can now emit a separate
workflow-action logit for fallback/verifier behavior. The current data is not
yet strong enough to replace the hand-selected margin gate. The head is
miscalibrated on held-out fallback opportunities and needs either:

- more fallback-opportunity labels, especially GLM-vs-DeepSeek cases, or
- a simpler low-dimensional calibration head over margin/entropy before using
the full sparse task feature set.

This should be treated as a prototype artifact and negative calibration result,
not a promoted policy.

## Next Step

Collect a targeted fallback-opportunity dataset where the top worker fails and
the second or third worker sometimes succeeds. Then train a fallback head with a
held-out calibration split before considering promotion.
