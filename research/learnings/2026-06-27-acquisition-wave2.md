# Acquisition Wave 2

## Question

After wave 1, can a specialist-only fresh-task pass add enough stable evidence
to move the repeated routing dataset closer to the 50-row M5 threshold without
hurting the active policy?

## Result

Only one stable row was added.

After excluding the active 23-task dataset, the wave1 merge-ready dataset, and
all prior outcome files, the canonical-pass top-four pool had only two fresh
specialist candidates left:

- `BigCodeBench/162`
- `BigCodeBench/208`

The repeated comparison ran 16 real worker outcomes across Qwen, Kimi, GLM, and
DeepSeek with two samples per worker/task. `BigCodeBench/162` was broad-pass
and became a Qwen latency target. `BigCodeBench/208` was all-fail and was
filtered out before merging.

The candidate stable dataset now has 33 rows:

- 23 active rows
- 9 wave1 merge-ready rows
- 1 wave2 merge-ready row

## Router Decision

The 33-task logits-router candidate was quarantined by the policy refresh gate.

Compared with the active 23-task baseline:

- LOO target accuracy fell from 0.783 to 0.636.
- LOO solvable pass@1 stayed at 0.800.
- LOO mean latency regret rose from 501 ms to 970 ms.

The candidate therefore adds useful measured evidence but should not replace
the active policy.

## Decision

Keep the wave2 row as measured data, but keep the active 23-task policy
unchanged. The current top-four canonical fresh pool is nearly exhausted under
the existing dependency profile. The next acquisition step should widen the
data source, either by adding a broader dependency profile or by moving a small
held-out slice from Terminal-Bench or another benchmark into measured routing
data.
