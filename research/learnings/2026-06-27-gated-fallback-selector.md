# Gated Fallback Threshold Selector

## Question

Can the conditional verifier/fallback behavior be selected by an auditable gate
instead of a hand-picked probability margin?

## Change

Added `tools/select_gated_fallback_threshold.py`. It sweeps candidate
top-two probability margins for the active logits router, requires every router
regression slice to pass, then ranks eligible margins by:

1. active solvable pass@1
2. active pass@1
3. active target accuracy
4. lower latency regret
5. lower mean latency
6. lower fallback rate

The selector writes:

- `research/refreshes/20260627-gated-fallback-threshold-selection.json`
- `research/policies/gated_fallback_policy.json`

The policy artifact is intentionally a workflow-policy candidate. It does not
replace `research/policies/active_policy.json`, because the active router is
still the base logits model.

## Result

The selected margin is `0.10` with `max_attempts=2`.

On the active 23-task dataset:

- solved: 19/23
- pass@1: 0.8261
- solvable pass@1: 0.95
- target accuracy: 0.9130
- fallbacks taken: 2 of 5 first-failure opportunities
- mean latency: 4886.9 ms
- mean latency regret: 748.3 ms

The selected policy passes both current regression slices:

- `heldout-hard-deepseek-763`
- `postrefresh-hard-glm-526`

Compared with stricter margins `0.02` and `0.05`, margin `0.10` adds one more
solved active task while keeping fallback use bounded. Looser margins preserve
the same solved count but add avoidable latency.

## Learning

The gate turns the verifier/fallback behavior into a selectable policy artifact
rather than an informal tuning knob. This is a useful intermediate step toward
the planned logits-emitting orchestrator: the future trainable head should
predict not only worker logits, but also whether the top worker is uncertain
enough to justify a verified fallback attempt.

The fresh post-refresh regression is still not fully solved as a first-route
learning problem. The gate rescues `BigCodeBench/526` through the second-ranked
DeepSeek attempt, while the empirical target remains GLM. That means the next
dataset/training step should add more GLM-vs-DeepSeek discriminating labels
rather than treating fallback success as routing success.

## Next Step

Promote this candidate only as a conditional workflow policy, then train or
prototype a small verifier/fallback logit head that predicts fallback probability
directly from task features and top-two router margin.
